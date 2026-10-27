#!/usr/bin/env python3
"""Rows of the per-pair benchmark table of the paper appendix, from results/per_pair.csv.

    python3 paper/make_table.py results/per_pair.csv
Prints LaTeX rows: sequence, method, attempt of success, tracked %, ATE [m], ATE [%], t_rel [%], r_rel [deg/100 m],
scale drift CV [%], initialisation delay [s]. Failed pairs show the best attempt and no error metrics.
"""
import sys
import pandas as pd

LABEL = {"orb2": "ORB", "sift_lightglue": "SIFT+LG", "akaze": "AKAZE", "superpoint_lightglue": "SP+LG"}
ORDER = list(LABEL)
d = pd.read_csv(sys.argv[1])
d["m"] = d.method.map(ORDER.index)
for seq, g in d.sort_values(["sequence", "m"]).groupby("sequence", sort=False):
    for k, (_, r) in enumerate(g.iterrows()):
        sid = seq[:2] if k == 0 else ""
        ok = r.success == 1
        att = f"{int(r.attempt_of_success)}" if ok else r"$\times$"
        f = (lambda v, n=2: f"{v:.{n}f}") if ok else (lambda v, n=2: "--")
        print(f"{sid} & {LABEL[r.method]} & {att} & {r.tracked_pct:.1f} & {f(r.ate_rmse_m)} & {f(r.ate_rmse_pct)} & "
              f"{f(r.kitti_t_rel_pct, 1)} & {f(r.kitti_r_rel_deg_100m, 1)} & {f(r.scale_drift_cv_pct, 1)} & "
              f"{r.init_delay_s:.1f}\\\\")
    print(r"\midrule")
