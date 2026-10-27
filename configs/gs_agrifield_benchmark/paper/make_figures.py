#!/usr/bin/env python3
"""Benchmark figures of the GS-AgriField paper (Figures 6, 7 and the trajectory examples).

    python3 paper/make_figures.py results <out_dir>
Reads results/per_pair.csv (written by aggregate.py) and results/trajectories_xy.csv
(written by paper/export_trajectories.py).
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.6,
                     "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.labelcolor": "#0b0b0b", "pdf.fonttype": 42})

METHODS = ["orb2", "sift_lightglue", "akaze", "superpoint_lightglue"]
LABEL = {"orb2": "ORB", "sift_lightglue": "SIFT+LightGlue", "akaze": "AKAZE", "superpoint_lightglue": "SuperPoint+LightGlue"}
COLOR = {"orb2": "#2a78d6", "sift_lightglue": "#eb6834", "akaze": "#1baf7a", "superpoint_lightglue": "#eda100"}
MARKER = {"orb2": "o", "sift_lightglue": "s", "akaze": "^", "superpoint_lightglue": "D"}
CROP = {"sugar beet": "sugar beet", "maize": "maize", "sunflower": "sunflower"}

d = pd.read_csv(os.path.join(src, "per_pair.csv"))
d["sid"] = d.sequence.str[:2]
seqs = sorted(d.sid.unique())
meta = d.groupby("sid").first()
ylab = [f"{s}  {meta.loc[s, 'crop']} {meta.loc[s, 'bbch']}, {meta.loc[s, 'speed']}" for s in seqs]

# 1. tracked frames per (sequence, method)
T = d.pivot(index="sid", columns="method", values="tracked_pct").loc[seqs, METHODS].fillna(0)
S = d.pivot(index="sid", columns="method", values="success").loc[seqs, METHODS].fillna(0)
A = d.pivot(index="sid", columns="method", values="attempt_of_success").loc[seqs, METHODS]
cmap = LinearSegmentedColormap.from_list("seq", ["#f4f3ee", "#9fc3ec", "#2a78d6", "#0d3a73"])
fig, ax = plt.subplots(figsize=(3.4, 3.1))
ax.imshow(T.values, cmap=cmap, vmin=0, vmax=100, aspect="auto")
for i in range(len(seqs)):
    for j, m in enumerate(METHODS):
        v, ok = T.iloc[i, j], S.iloc[i, j] == 1
        txt = f"{v:.0f}" + ("" if not ok or A.iloc[i, j] == 1 else f" ({A.iloc[i, j]:.0f})") if ok else f"{v:.0f} ✗"
        ax.text(j, i, txt, ha="center", va="center", fontsize=7, color="white" if v > 60 else "#0b0b0b",
                fontweight="normal" if ok else "bold")
        ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="white", lw=1.5))
ax.set_xticks(range(len(METHODS)), [LABEL[m].replace("+", "+\n") for m in METHODS], fontsize=7)
ax.set_yticks(range(len(seqs)), ylab, fontsize=7)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(out, f"tracking_success.{ext}"), dpi=300)
plt.close(fig)

# 2. ATE (% of path length) per sequence, failures marked on the right
fig, ax = plt.subplots(figsize=(3.4, 3.0))
off = dict(zip(METHODS, np.linspace(-0.27, 0.27, 4)))
for m in METHODS:
    g = d[d.method == m].set_index("sid").loc[seqs]
    y = np.arange(len(seqs)) + off[m]
    ok = g.success.values == 1
    ax.scatter(g.ate_rmse_pct.values[ok], y[ok], s=18, marker=MARKER[m], color=COLOR[m], ec="white", lw=0.5,
               label=LABEL[m], zorder=3)
    ax.scatter(np.full((~ok).sum(), 2.75), y[~ok], s=22, marker="x", color=COLOR[m], lw=1.2, zorder=3)
ax.axvline(2.6, color="#c3c2b7", lw=0.6)
ax.text(2.75, -0.9, "failed", ha="center", fontsize=7, color="#52514e")
ax.set_xlim(0.3, 2.95)
ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5])
ax.set_ylim(len(seqs) - 0.5, -1.2)
ax.set_yticks(range(len(seqs)), ylab, fontsize=7)
ax.set_xlabel("ATE RMSE [% of path length]")
ax.grid(axis="x", color="#e6e5e0", lw=0.5)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.17), ncol=2, frameon=False, fontsize=7, handletextpad=0.3)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(out, f"ate_per_sequence.{ext}"), dpi=300)
plt.close(fig)

# 3. top-view trajectories of three representative sequences
tr = pd.read_csv(os.path.join(src, "trajectories_xy.csv"))
show = [(s, f"({c}) {s[:2]} {meta.loc[s[:2], 'crop']} {meta.loc[s[:2], 'bbch']}")
        for c, s in zip("abc", ["01_sugarbeet_bbch19p_fast_run001", "07_maize_bbch18_fast_run001",
                                "08_sunflower_bbch10_fast_run001"])]
fig, axs = plt.subplots(1, 3, figsize=(7.0, 3.5))
for ax, (seq, title) in zip(axs, show):
    g = tr[tr.sequence == seq]
    r = g[g.kind == "reference"]
    x0, y0 = r.x.min(), r.y.min()
    ax.plot(r.x - x0, r.y - y0, color="#0b0b0b", lw=2.2, label="RTK-GNSS/INS reference", zorder=2)
    for m in METHODS:
        e = g[(g.kind == "estimate") & (g.method == m)]
        p = d[(d.sequence == seq) & (d.method == m)].iloc[0]
        if len(e) and p.success == 1:
            ax.plot(e.x - x0, e.y - y0, color=COLOR[m], lw=1.0, label=LABEL[m], zorder=3)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title, fontsize=8, loc="left")
    ax.set_xlabel("x [m]")
    ax.grid(color="#e6e5e0", lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axs[0].set_ylabel("y [m]")
h, l = [], []
for ax in axs:
    for hh, ll in zip(*ax.get_legend_handles_labels()):
        if ll not in l:
            h.append(hh); l.append(ll)
fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=7)
fig.tight_layout(rect=(0, 0.07, 1, 1))
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(out, f"trajectories_examples.{ext}"), dpi=300)
plt.close(fig)
print("figures written to", out)
