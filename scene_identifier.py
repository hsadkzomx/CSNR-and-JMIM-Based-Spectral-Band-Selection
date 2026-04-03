import os
import glob
import argparse
import numpy as np
import pandas as pd
import imageio.v3 as iio
import matplotlib.pyplot as plt


def load_rgb(path):
    img = iio.imread(path)

    if img.ndim == 2:  # grayscale -> 3ch
        img = np.stack([img, img, img], axis=-1)

    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError(f"Unexpected image shape for {path}: {img.shape}")

    img = img[..., :3]  # drop alpha if any

    if np.issubdtype(img.dtype, np.integer):
        maxv = np.iinfo(img.dtype).max
        img = img.astype(np.float32) / float(maxv)
    else:
        img = img.astype(np.float32)

    # HDR pipelines can have negatives; clip for luminance computation
    img = np.clip(img, 0.0, None)
    return img


def compute_scene_score(
    rgb,
    *,
    score_quantile = 0.5,   # 0.5=median; use 0.2 to emphasize dark regions
    winsor_top_pct = 99.5,  # trims headlights/outliers (set None to disable)
    use_log = False,         # optional HDR/perceptual compression
    eps_rel = 1e-6,         # epsilon for log; relative to scene max luminance
    sample_stride = 1         # set 2/4 for speed; keep fixed for reproducibility
):
    
    if sample_stride > 1:
        rgb = rgb[::sample_stride, ::sample_stride, :]

    # Luminance in linear domain
    Y = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(np.float32)
    Y = Y[np.isfinite(Y)]
    if Y.size == 0:
        return np.nan

    if use_log:
        ymax = float(np.max(Y))
        eps = eps_rel * ymax if ymax > 0 else eps_rel
        Y = np.log(Y + eps).astype(np.float32)

    if winsor_top_pct is not None:
        hi = float(np.percentile(Y, winsor_top_pct))
        Y = np.minimum(Y, hi)

    return float(np.quantile(Y, score_quantile))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--patterns", nargs="+", default=["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp", "*.exr"])
    ap.add_argument("--q_bottom", type=float, default=0.20)
    ap.add_argument("--score_quantile", type=float, default=0.5)
    ap.add_argument("--winsor_top_pct", type=float, default=99.5)
    ap.add_argument("--no_winsor", action="store_true")
    ap.add_argument("--use_log", action="store_true")
    ap.add_argument("--sample_stride", type=int, default=1)
    ap.add_argument("--out_csv", default="illum_scores.csv")
    ap.add_argument("--out_hist", default="illum_hist.png")
    ap.add_argument("--out_ids", default="low_illum_scene_ids.txt")
    args = ap.parse_args()

    # Collect paths
    paths = []
    for pat in args.patterns:
        paths.extend(glob.glob(os.path.join(args.img_dir, pat)))
    paths = sorted(paths)

    if not paths:
        raise FileNotFoundError(f"No images found in {args.img_dir} with patterns {args.patterns}")

    winsor = None if args.no_winsor else args.winsor_top_pct

    rows = []
    for p in paths:
        scene_id = os.path.basename(p)
        rgb = load_rgb(p)
        s = compute_scene_score(
            rgb,
            score_quantile=args.score_quantile,
            winsor_top_pct=winsor,
            use_log=args.use_log,
            sample_stride=args.sample_stride
        )
        rows.append((scene_id, s))

    df = pd.DataFrame(rows, columns=["scene_id", "S_illum"])
    df = df.dropna().sort_values("S_illum").reset_index(drop=True)

    # Rank-based selection to guarantee exactly ceil(q% * N) scenes.
    n_low = int(np.ceil(args.q_bottom * len(df)))
    df["low_illum"] = np.arange(len(df)) < n_low
    cutoff = float(df.loc[n_low - 1, "S_illum"]) if n_low > 0 else float("nan")

    # Save CSV (auditable)
    df.to_csv(args.out_csv, index=False)

    # Save selected IDs
    low_ids = df.loc[df["low_illum"], "scene_id"].tolist()
    with open(args.out_ids, "w", encoding="utf-8") as f:
        for sid in low_ids:
            f.write(sid + "\n")

    # Histogram
    plt.figure()
    plt.hist(df["S_illum"].values, bins=50)
    if np.isfinite(cutoff):
        plt.axvline(cutoff, linewidth=2, color="red")
    # plt.title(
    #     f"Scene illumination scores (N={len(df)}), low-illumination stress subset={n_low}/{len(df)}"
    # )
    plt.xlabel("S_illum (scene score)")
    plt.ylabel("Number of scenes")
    plt.tight_layout()
    plt.savefig(args.out_hist, dpi=200)
    plt.close()

    print(f"Processed scenes: {len(df)}")
    print(f"Cutoff score (rank-based, q={args.q_bottom}): {cutoff}")
    print(f"Selected low-illumination stress subset scenes: {df['low_illum'].sum()}")
    print(f"Wrote: {args.out_csv}, {args.out_hist}, {args.out_ids}")


if __name__ == "__main__":
    main()
