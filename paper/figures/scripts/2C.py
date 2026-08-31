"""Global DockQ by MHC class (I vs II), one box per class, dots coloured per model.

Data: output/DockQ/dockq_all.csv (global_dockq repeats per interface -> collapsed per pdb).
Gray-area entries are dropped here, at analysis time (CLAUDE.md).
"""
import glob, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/14TBDrive/6TBDrive1_backup/benchmark_fresh"
OUT = f"{BASE}/paper/figures/outputs"

MODELS = ["AF3", "Protenix", "ESMFold2"]
from style import COLORS, BASE_COLORS, keep, load_dockq, global_dockq
CLASSES = ["Class I", "Class II"]
BANDS = [(0.00, 0.23, "#f7dfe0", "Incorrect", "#b5545c"),
         (0.23, 0.49, "#fdf1dd", "Acceptable", "#d38b2a"),
         (0.49, 0.80, "#e4f1fa", "Medium", "#3f86c4"),
         (0.80, 1.05, "#e7e2f3", "High", "#5a4a9c")]

plt.rcParams.update({"font.size": 13, "axes.labelsize": 14,
                     "xtick.labelsize": 12, "ytick.labelsize": 11,
                     "savefig.dpi": 200, "figure.dpi": 200})

d = global_dockq(keep(load_dockq()), ["mhc_class"])
d = d[d.model.isin(MODELS)]        # the boxes aggregate every row, so keep them to this panel
assert set(d.mhc_class) <= set(CLASSES), sorted(set(d.mhc_class))

fig, ax = plt.subplots(figsize=(6.0, 5.6))
for lo, hi, fill, lab, tc in BANDS:
    ax.axhspan(lo, hi, color=fill, zorder=0)
    ax.text(0.985, (lo + min(hi, 1.0)) / 2, lab, transform=ax.get_yaxis_transform(),
            ha="right", va="center", fontsize=10.5, color=tc, fontweight="bold")

groups = [d[d.mhc_class == c].global_dockq.values for c in CLASSES]
ax.boxplot(groups, positions=[1, 2], widths=.55, showfliers=False, zorder=2,
           medianprops=dict(color="black", lw=2.2),
           boxprops=dict(color="black", lw=1.4, facecolor="white", alpha=.55),
           whiskerprops=dict(color="black", lw=1.4), capprops=dict(color="black", lw=1.4),
           patch_artist=True)

rng = np.random.default_rng(0)  # ponytail: fixed seed so the jitter is reproducible
for m in MODELS:
    sub = d[d.model == m]
    if not len(sub):
        continue
    x = np.array([CLASSES.index(c) + 1 for c in sub.mhc_class], float)
    ax.scatter(x + rng.uniform(-.17, .17, len(x)), sub.global_dockq, s=34,
               facecolor=COLORS[m], edgecolor="black", linewidth=.6, alpha=.9,
               zorder=3, label=f"{m} (n={len(sub)})")

for i, (c, g) in enumerate(zip(CLASSES, groups)):
    if len(g):
        ax.text(i + 1, 1.06, f"med {np.median(g):.2f}\nn={len(g)}",
                ha="center", va="bottom", fontsize=10.5)

ax.set_title(r"$\bf{B}$", loc="left", fontsize=17, pad=6)
ax.set_xticks([1, 2]); ax.set_xticklabels(CLASSES)
ax.set_xlim(.4, 2.6); ax.set_ylim(-0.02, 1.02)
ax.set_ylabel(r"Global DockQ")
ax.legend(title="Model", loc="upper center", bbox_to_anchor=(.5, -.10),
          ncol=min(3, max(1, d.model.nunique())), frameon=True, fontsize=11, title_fontsize=12)

os.makedirs(OUT, exist_ok=True)
fig.savefig(f"{OUT}/Figure_2C.png", bbox_inches="tight")
plt.close(fig)
print(f"{OUT}/Figure_2C.png  " +
      ", ".join(f"{c}:{len(g)}" for c, g in zip(CLASSES, groups)))
