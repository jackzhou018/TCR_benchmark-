"""Figure 1B: CAPRI classification of Global DockQ, one 100% stacked bar per model.

Data: output/DockQ/dockq_all.csv via style.load_dockq/keep/global_dockq, so the n per model
matches every other panel (CLAUDE.md). Layout and colours follow paper/figures/templates/1C.png;
the class cuts are the same ones the BANDS in 1B/2C use (0.23 / 0.49 / 0.80).
"""
import os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from style import MODELS, keep, load_dockq, global_dockq

# Same column order as 1A. TCRmodel2 is class I only, so its bar is over 111 structures,
# not 126 -- the n above each bar says so.
PANEL_MODELS = MODELS + ["TCRmodel2"]

BASE = "/14TBDrive/6TBDrive1_backup/benchmark_fresh"
OUT = f"{BASE}/paper/figures/outputs"

# bottom of the stack first. Colours sampled from templates/1C.png.
CLASSES = [("High (0.8-1)",           0.80, 1.01, "#2818E1"),
           ("Medium (0.49-0.8)",      0.49, 0.80, "#65C0F8"),
           ("Acceptable (0.23-0.49)", 0.23, 0.49, "#F6CA72"),
           ("Incorrect (0-0.23)",     0.00, 0.23, "#A1342F")]

plt.rcParams.update({"font.size": 13, "axes.labelsize": 14,
                     "xtick.labelsize": 11, "ytick.labelsize": 11,
                     "savefig.dpi": 200, "figure.dpi": 200})

d = global_dockq(keep(load_dockq()))
models = [m for m in PANEL_MODELS if (d.model == m).any()]

fig, ax = plt.subplots(figsize=(6.6, 7.0))
x = np.arange(len(models))
bottom = np.zeros(len(models))

for label, lo, hi, colour in CLASSES:
    pct = np.array([100 * ((s >= lo) & (s < hi)).mean()
                    for s in (d[d.model == m].global_dockq for m in models)])
    ax.bar(x, pct, bottom=bottom, width=.62, color=colour,
           edgecolor="black", linewidth=.9, label=label, zorder=2)
    for xi, (p, b) in enumerate(zip(pct, bottom)):
        if p > 0:
            ax.text(xi, b + p / 2, f"{p:.1f}%", ha="center", va="center",
                    rotation=60, fontsize=9.5, zorder=3)
    bottom += pct

assert np.allclose(bottom, 100), bottom   # every structure lands in exactly one class

for xi, m in enumerate(models):
    ax.text(xi, 101, f"n={(d.model == m).sum()}", ha="center", va="bottom", fontsize=11)

if not models:
    ax.text(.5, .5, "no DockQ results yet", transform=ax.transAxes,
            ha="center", color="gray", fontsize=11)

ax.set_ylabel("Percentage")
ax.set_ylim(0, 100); ax.set_yticks(range(0, 101, 10))
ax.set_xticks(x); ax.set_xticklabels(models, rotation=45, ha="right")
ax.set_xlim(-.55, len(models) - .45)
ax.set_title(r"$\bf{B}$", loc="left", fontsize=17, pad=26)
# legend top-down: High first, i.e. the reverse of the draw order
leg = ax.legend(handles=[Patch(facecolor=c, edgecolor="black", label=l) for l, _, _, c in CLASSES],
                title="CAPRI\nClassification (DockQ)", loc="center left", bbox_to_anchor=(1.02, .5),
                frameon=True, fontsize=11, title_fontsize=12)
leg.get_title().set_multialignment("center")   # else "CAPRI" hangs off the left

os.makedirs(OUT, exist_ok=True)
fig.savefig(f"{OUT}/Figure_1B.png", bbox_inches="tight")
plt.close(fig)
print(f"{OUT}/Figure_1B.png  (" + ", ".join(f"{m}:{(d.model == m).sum()}" for m in models) + ")")
