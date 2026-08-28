#!/usr/bin/env python
"""AF3: pDockQ, pDockQ2 and ipTM against Global DockQ, same axes and CAPRI bands.

    python pDockQ/pdockq2/plot_pdockq2.py   -> results/pdockq2_vs_dockq_AF3.png

AF3 only -- pDockQ2 needs the PAE matrix and no other model kept one (see run_pdockq2.py).
"""
import os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr, pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, f"{ROOT}/paper/figures/scripts")
import style

BANDS = [(0.00, 0.23, "#f7dfe0", "Incorrect", "#b5545c"),
         (0.23, 0.49, "#fdf1dd", "Acceptable", "#d38b2a"),
         (0.49, 0.80, "#e4f1fa", "Medium", "#3f86c4"),
         (0.80, 1.05, "#e7e2f3", "High", "#5a4a9c")]
COLOR = style.BASE_COLORS["AF3"]

d = style.keep(pd.read_csv(f"{HERE}/results/pdockq2_per_structure.csv")).dropna(
    subset=["global_dockq"])
v1 = style.keep(pd.read_csv(f"{os.path.dirname(HERE)}/results/pdockq_per_structure.csv"))
v1 = v1[v1.model == "AF3"][["pdb_id", "mean_pdockq"]]
iptm = pd.read_csv(f"{ROOT}/models/AF3/results/af3-server_results.csv")[["pdb_id", "iptm"]]
af3 = d.merge(v1, on="pdb_id").merge(iptm, on="pdb_id")


def panel(ax, x, y, xlabel, title):
    for lo, hi, fill, lab, tc in BANDS:
        ax.axhspan(lo, hi, color=fill, zorder=0)
    ax.scatter(x, y, s=30, color=style.lighten(COLOR), edgecolor=COLOR, linewidth=.7, zorder=3)
    rp, pp = pearsonr(x, y)
    rs, _ = spearmanr(x, y)
    ax.set_title(f"{title}\nPearson r={rp:.2f}   Spearman $\\rho$={rs:.2f}   p={pp:.1e}",
                 fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=.25, zorder=1)


fig, axes = plt.subplots(1, 3, figsize=(14, 5.2), sharey=True)
panel(axes[0], af3.mean_pdockq, af3.global_dockq, "mean pDockQ over chain pairs", "pDockQ")
panel(axes[1], af3.mean_pdockq2, af3.global_dockq, "mean pDockQ2 over chains", "pDockQ2")
panel(axes[2], af3.iptm, af3.global_dockq, "ipTM", "ipTM")
axes[0].set_ylabel("Global DockQ (measured vs native)")
for lo, hi, fill, lab, tc in BANDS:
    axes[0].text(0.02, (lo + min(hi, 1.0)) / 2, lab, transform=axes[0].get_yaxis_transform(),
                 ha="left", va="center", fontsize=10, color=tc, fontweight="bold")
fig.suptitle(f"AF3: predicting DockQ without the native   n={len(af3)}", fontsize=13)
fig.tight_layout()
out = f"{HERE}/results/pdockq2_vs_dockq_AF3.png"
fig.savefig(out, dpi=200)
print(out)

for lab, col in (("pDockQ ", "mean_pdockq"), ("pDockQ2", "mean_pdockq2"), ("ipTM   ", "iptm")):
    print(f"{lab} vs DockQ : Pearson {pearsonr(af3[col], af3.global_dockq)[0]:.2f}  "
          f"Spearman {spearmanr(af3[col], af3.global_dockq)[0]:.2f}  "
          f"median {af3[col].median():.3f}")
