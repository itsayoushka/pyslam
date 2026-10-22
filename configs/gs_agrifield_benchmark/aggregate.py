#!/usr/bin/env python3
"""Aggregate runs.csv (evaluate.py) under the retry-until-success protocol.

    .venv_eval/bin/python aggregate.py [--dir results]

Per (sequence, method): the reported run is the FIRST successful attempt (never the lowest-error one); if no
attempt succeeded, the attempt with the most tracked frames is reported as a failure. Outputs in <dir>/:
  per_pair.csv            reported run + attempts used + first-attempt outcome, for every (sequence, method)
  table_per_sequence.*    tracked % / ATE % per sequence and method (CSV, Markdown, LaTeX)
  table_summary.*         per method: success within <=3 attempts, first-attempt success, attempts, median errors
  table_by_crop.*, table_by_illumination.*   same summary grouped by crop / illumination label
  fig_success_heatmap.*, fig_ate_box.*       figures (PNG + PDF)
Errors (ATE, RPE, drift, scale) are aggregated over successful pairs only; success rates over all pairs.
"""
import argparse, os
import numpy as np
import pandas as pd

B = os.path.dirname(os.path.abspath(__file__))
METHOD_ORDER = ["orb2", "sift", "sift_lightglue", "akaze", "superpoint_lightglue", "xfeat"]
LABEL = {"orb2": "ORB", "sift": "SIFT", "sift_lightglue": "SIFT+LightGlue", "akaze": "AKAZE", "superpoint_lightglue": "SuperPoint+LightGlue", "xfeat": "XFeat"}
ERR = {"ate_rmse_m": "ATE RMSE [m]", "ate_rmse_pct": "ATE [% length]", "rpe_t_1m_m": "RPE$_t$ 1 m [m]",
       "rpe_r_1m_deg": "RPE$_r$ 1 m [deg]", "kitti_t_rel_pct": "$t_{rel}$ [%]",
       "kitti_r_rel_deg_100m": "$r_{rel}$ [deg/100 m]", "scale_drift_cv_pct": "scale drift CV [%]"}


def reported_runs(df):
    out = []
    for (seq, m), g in df.groupby(["sequence", "method"]):
        g = g.sort_values("rep")
        ok = g[g["success"] == 1]
        r = (ok.iloc[0] if len(ok) else g.sort_values(["tracked_pct", "rep"], ascending=[False, True]).iloc[0]).copy()
        r["attempts_run"] = len(g)
        r["attempt_of_success"] = int(ok.iloc[0]["rep"]) if len(ok) else np.nan
        r["first_attempt_success"] = int(g.iloc[0]["success"] == 1)
        out.append(r)
    return pd.DataFrame(out)


def save(df, base, caption):
    df.to_csv(base + ".csv", index=False)
    cols = list(df.columns)
    fmt = lambda v: f"{v:.2f}" if isinstance(v, (float, np.floating)) and not np.isnan(v) else ("--" if isinstance(v, float) else str(v))
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    open(base + ".md", "w").write("\n".join(md) + "\n")
    tex = ["\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}", "\\small",
           "\\begin{tabular}{" + "l" * 1 + "r" * (len(cols) - 1) + "}", "\\toprule",
           " & ".join(c.replace("_", "\\_").replace("%", "\\%") for c in cols) + " \\\\", "\\midrule"]
    tex += [" & ".join(fmt(v).replace("_", "\\_").replace("%", "\\%") for v in r) + " \\\\" for r in df.itertuples(index=False)]
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(base + ".tex", "w").write("\n".join(tex) + "\n")


def summary(rep, by=None):
    keys = ([by] if by else []) + ["method"]
    rows = []
    for k, g in rep.groupby(keys):
        k = k if isinstance(k, tuple) else (k,)
        s = g[g["success"] == 1]
        row = dict(zip(keys, k))
        row.update(pairs=len(g), success=int(g["success"].sum()),
                   success_pct=100 * g["success"].mean(), first_try_pct=100 * g["first_attempt_success"].mean(),
                   mean_attempts=g["attempts_run"].mean(), tracked_pct=g["tracked_pct"].mean())
        for c in ERR:
            row[c] = s[c].median() if c in s and len(s) else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    out["method"] = pd.Categorical(out["method"], [m for m in METHOD_ORDER if m in set(out["method"])])
    out = out.sort_values(keys[:-1] + ["method"]).reset_index(drop=True)
    out["method"] = out["method"].astype(str).map(LABEL)
    return out


def figures(rep, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = [m for m in METHOD_ORDER if m in set(rep["method"])]
    seqs = sorted(rep["sequence"].unique())
    M = np.full((len(seqs), len(methods)), np.nan)
    A = np.full_like(M, np.nan)
    for i, s in enumerate(seqs):
        for j, m in enumerate(methods):
            r = rep[(rep.sequence == s) & (rep.method == m)]
            if len(r):
                M[i, j], A[i, j] = r["tracked_pct"].iloc[0], r["attempt_of_success"].iloc[0]
    fig, ax = plt.subplots(figsize=(1.3 * len(methods) + 2.5, 0.38 * len(seqs) + 1.4))
    im = ax.imshow(M, vmin=0, vmax=100, cmap="RdYlGn", aspect="auto")
    for i in range(len(seqs)):
        for j in range(len(methods)):
            if not np.isnan(M[i, j]):
                tag = f"{M[i, j]:.0f}" + ("" if np.isnan(A[i, j]) else f" ({int(A[i, j])})")
                ax.text(j, i, tag if not np.isnan(A[i, j]) else f"{M[i, j]:.0f} ✗", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(methods)), [LABEL[m] for m in methods], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(seqs)), seqs, fontsize=7)
    fig.colorbar(im, ax=ax, label="tracked frames [%]")
    ax.set_title("tracked % (attempt of first success; ✗ = failed after all attempts)", fontsize=8)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"fig_success_heatmap.{e}"), dpi=200)
    plt.close(fig)
    s = rep[rep.success == 1]
    if len(s):
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        data = [s[s.method == m]["ate_rmse_pct"].dropna() for m in methods]
        ax.boxplot(data, showfliers=True)
        ax.set_xticks(range(1, len(methods) + 1), [f"{LABEL[m]}\n(n={len(d)})" for m, d in zip(methods, data)], fontsize=7)
        ax.set_ylabel("ATE RMSE [% of path length]"); ax.grid(alpha=.3, axis="y")
        ax.set_title("successful runs only", fontsize=8)
        fig.tight_layout()
        for e in ("png", "pdf"):
            fig.savefig(os.path.join(out, f"fig_ate_box.{e}"), dpi=200)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(B, "results"))
    a = ap.parse_args()
    df = pd.read_csv(os.path.join(a.dir, "runs.csv"))
    df["success"] = df.get("success", 0).fillna(0).astype(int)
    rep = reported_runs(df)
    rep.to_csv(os.path.join(a.dir, "per_pair.csv"), index=False)
    per_seq = rep.pivot_table(index="sequence", columns="method", values="tracked_pct").reindex(
        columns=[m for m in METHOD_ORDER if m in set(rep.method)]).rename(columns=LABEL).reset_index()
    save(per_seq, os.path.join(a.dir, "table_per_sequence_tracked"), "Tracked frames (\\%) per sequence (reported run).")
    ate = rep[rep.success == 1].pivot_table(index="sequence", columns="method", values="ate_rmse_pct").reindex(
        columns=[m for m in METHOD_ORDER if m in set(rep.method)]).rename(columns=LABEL).reset_index()
    save(ate, os.path.join(a.dir, "table_per_sequence_ate"), "ATE after Sim(3) alignment (\\% of path length), successful runs.")
    save(summary(rep), os.path.join(a.dir, "table_summary"), "Baseline summary over all valid sequences.")
    save(summary(rep, "crop"), os.path.join(a.dir, "table_by_crop"), "Baseline summary per crop.")
    save(summary(rep, "illumination"), os.path.join(a.dir, "table_by_illumination"),
         "Baseline summary per illumination label (confounded with acquisition date).")
    figures(rep, a.dir)
    print(open(os.path.join(a.dir, "table_summary.md")).read())


if __name__ == "__main__":
    main()
