import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def overlap_rate(a, b):
    # overlap of A relative to baseline B
    if not b:
        return 1.0
    return len(a & b) / len(b)


def read_ids(txt_path):
    return set([x.strip() for x in txt_path.read_text(encoding="utf-8").splitlines() if x.strip()])


def spearman_rank_corr(csv_a, csv_b):
    da = pd.read_csv(csv_a)[["scene_id", "S_illum"]].set_index("scene_id")
    db = pd.read_csv(csv_b)[["scene_id", "S_illum"]].set_index("scene_id")
    d = da.join(db, how="inner", lsuffix="_a", rsuffix="_b").dropna()
    if len(d) < 3:
        return float("nan")
    ra = d["S_illum_a"].rank(method="average")
    rb = d["S_illum_b"].rank(method="average")
    return float(ra.corr(rb))


def run_one(
    py_exe,
    selector_script,
    img_dir,
    patterns,
    q_bottom,
    score_quantile,
    winsor_top_pct,
    use_log,
    sample_stride,
    outdir,
    tag,
):
    outdir.mkdir(parents=True, exist_ok=True)

    out_csv = outdir / f"illum_scores__{tag}.csv"
    out_hist = outdir / f"illum_hist__{tag}.png"
    out_ids = outdir / f"low_illum_scene_ids__{tag}.txt"

    cmd = [
        py_exe,
        str(selector_script),
        "--img_dir", str(img_dir),
        "--q_bottom", str(q_bottom),
        "--score_quantile", str(score_quantile),
        "--sample_stride", str(sample_stride),
        "--out_csv", str(out_csv),
        "--out_hist", str(out_hist),
        "--out_ids", str(out_ids),
    ]

    # patterns (repeatable arg list)
    cmd += ["--patterns"] + patterns

    # winsor control
    if winsor_top_pct is None:
        cmd += ["--no_winsor"]
    else:
        cmd += ["--winsor_top_pct", str(winsor_top_pct)]

    # log control
    if use_log:
        cmd += ["--use_log"]

    # run
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"Run failed for tag={tag}\nCMD:\n{cmd}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"
        )

    # parse outputs
    df = pd.read_csv(out_csv)
    selected_ids = set(df.loc[df["low_illum"] == True, "scene_id"].tolist())
    cutoff = float(df.loc[df["low_illum"] == True, "S_illum"].max()) if len(selected_ids) else float("nan")

    return {
        "tag": tag,
        "q_bottom": q_bottom,
        "score_quantile": score_quantile,
        "winsor_top_pct": winsor_top_pct,
        "use_log": use_log,
        "sample_stride": sample_stride,
        "n_total": int(len(df)),
        "n_selected": int(len(selected_ids)),
        "cutoff": cutoff,
        "out_csv": str(out_csv),
        "out_ids": str(out_ids),
        "out_hist": str(out_hist),
        "stdout": p.stdout.strip(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selector_script", required=True)
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--patterns", nargs="+", default=["*.jpg"])

    ap.add_argument("--out_root", default="illum_sensitivity_runs")
    ap.add_argument("--py", default=sys.executable)

    # Baseline run
    ap.add_argument("--base_q_bottom", type=float, default=0.20)
    ap.add_argument("--base_score_quantile", type=float, default=0.50)
    ap.add_argument("--base_winsor_top_pct", type=float, default=99.5)
    ap.add_argument("--base_use_log", action="store_true")
    ap.add_argument("--base_sample_stride", type=int, default=1)

    # Which sensitivity(s) to run
    ap.add_argument("--run_winsor", action="store_true")
    ap.add_argument("--run_quantile", action="store_true")
    ap.add_argument("--run_qbottom", action="store_true")

    args = ap.parse_args()

    selector_script = Path(args.selector_script)
    img_dir = Path(args.img_dir)
    out_root = Path(args.out_root)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = out_root / stamp
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- 0) Baseline run ----
    base_tag = "BASE"
    base = run_one(
        py_exe=args.py,
        selector_script=selector_script,
        img_dir=img_dir,
        patterns=args.patterns,
        q_bottom=args.base_q_bottom,
        score_quantile=args.base_score_quantile,
        winsor_top_pct=args.base_winsor_top_pct,
        use_log=args.base_use_log,
        sample_stride=args.base_sample_stride,
        outdir=outdir,
        tag=base_tag,
    )
    base_ids = read_ids(Path(base["out_ids"]))

    results = []

    def add_run(tag, q_bottom, score_quantile, winsor_top_pct):
        r = run_one(
            py_exe=args.py,
            selector_script=selector_script,
            img_dir=img_dir,
            patterns=args.patterns,
            q_bottom=q_bottom,
            score_quantile=score_quantile,
            winsor_top_pct=winsor_top_pct,
            use_log=args.base_use_log,
            sample_stride=args.base_sample_stride,
            outdir=outdir,
            tag=tag,
        )
        ids = read_ids(Path(r["out_ids"]))
        r["jaccard_vs_base"] = jaccard(ids, base_ids)
        r["overlap_vs_base"] = overlap_rate(ids, base_ids)
        r["n_changed_vs_base"] = int(len(ids ^ base_ids))
        r["spearman_rho_vs_base"] = spearman_rank_corr(Path(r["out_csv"]), Path(base["out_csv"]))
        results.append(r)

    # Always include baseline in summary
    base_summary = dict(base)
    base_summary["jaccard_vs_base"] = 1.0
    base_summary["overlap_vs_base"] = 1.0
    base_summary["n_changed_vs_base"] = 0
    base_summary["spearman_rho_vs_base"] = 1.0
    results.append(base_summary)

    # ---- 1) Winsorization sensitivity ----
    if args.run_winsor:
        sweep = [None, 99.0, 99.5, 99.9]
        for w in sweep:
            tag = f"WINSOR_{'NONE' if w is None else str(w)}"
            add_run(tag, args.base_q_bottom, args.base_score_quantile, w)

    # ---- 2) Quantile pooling sensitivity ----
    if args.run_quantile:
        sweep = [0.30, 0.50, 0.70]
        for q in sweep:
            tag = f"QPOOL_{q}"
            add_run(tag, args.base_q_bottom, q, args.base_winsor_top_pct)

    # ---- 3) q-bottom sensitivity (optional) ----
    if args.run_qbottom:
        sweep = [0.10, 0.20, 0.30]
        for qb in sweep:
            tag = f"QBOTTOM_{qb}"
            add_run(tag, qb, args.base_score_quantile, args.base_winsor_top_pct)

    # ---- Save logs ----
    summary_csv = outdir / "illum_sensitivity_summary.csv"
    pd.DataFrame(results).sort_values("tag").to_csv(summary_csv, index=False)

    cfg_json = outdir / "run_config.json"
    cfg_json.write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    print("\n=== DONE ===")
    print(f"Output folder: {outdir}")
    print(f"Summary CSV:   {summary_csv}")


if __name__ == "__main__":
    main()
