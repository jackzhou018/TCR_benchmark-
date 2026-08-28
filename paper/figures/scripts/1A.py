"""Figures for the fresh TCR-pMHC benchmark (136 post-cutoff entries).

Adapted from the ImmunoBench figure script. Everything ImmunoBench-specific is gone:
no training-cutoff or redundancy filtering (the whole set is post-cutoff by construction),
no TCRdock/Boltz-2, and no ESMFold2 fast/full split beyond the configs actually on disk.

Data contract: the long per-interface CSVs written by scripts/run_dockq.py, collapsed to one
Global DockQ per structure by style.global_dockq() -- the same metric every other panel plots.
"""
import glob, os, re, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns

BASE = "/14TBDrive/6TBDrive1_backup/benchmark_fresh"
OUT = f"{BASE}/paper/figures/outputs"

COLW = 5.5 / 3          # per-column width, kept from the 3-model figure
MODELS = ["AF3", "Protenix", "ESMFold2", "TCRmodel2"]    # models/<MODEL>/, plot order
from style import COLORS, BASE_COLORS, keep, load_dockq, global_dockq   # shared palette + metric
plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13,
                     "xtick.labelsize": 11, "ytick.labelsize": 11,
                     "savefig.dpi": 200, "figure.dpi": 200})

def load():
    """One row per (label, global_dockq). Label = model, or model-config if >1 config.

    Gray-area (REVIEW) and unscoreable-native rows are dropped by style.keep(), at analysis
    time, not at input time (CLAUDE.md).
    """
    d = global_dockq(keep(load_dockq()))
    rows, order = [], []
    for m in MODELS:
        sub = d[d.model == m]
        cfgs = sorted(sub.config.unique())
        for cfg in cfgs:
            label = m if len(cfgs) == 1 else f"{m}-{cfg}"
            order.append(label)
            COLORS.setdefault(label, COLORS[m])
            rows += [{"Method": label, "val": v} for v in sub[sub.config == cfg].global_dockq]
        if not cfgs:                                     # keep an empty slot on the axis
            order.append(m)
    return pd.DataFrame(rows, columns=["Method", "val"]), order


def strip_box(ax, data, order, ylabel, title):
    if len(data):
        sns.violinplot(data=data, x="Method", y="val", order=order, hue="Method", legend=False,
                       palette=[COLORS[m] for m in order], inner=None, alpha=.6, ax=ax, cut=0)
        sns.boxplot(data=data, x="Method", y="val", order=order, width=.2, showcaps=False,
                    boxprops={"facecolor": "none", "edgecolor": "black"}, showfliers=False,
                    whiskerprops={"linewidth": 1.5, "color": "black"},
                    medianprops={"color": "black", "linewidth": 2}, ax=ax)
        sns.stripplot(data=data, x="Method", y="val", order=order, color="black", size=2.5,
                      alpha=.5, jitter=.22, ax=ax)
    else:
        ax.set_xticks(range(len(order)), order)
        ax.text(.5, .5, "no DockQ results yet", transform=ax.transAxes, ha="center",
                color="gray", fontsize=11)
    n_full = data.Method.value_counts().max() if len(data) else 0
    for i, m in enumerate(order):                      # median above each violin
        v = data[data.Method == m].val if len(data) else []
        if len(v):
            # A column short of the full signed-off set says so. TCRmodel2 has all
            # runnable Class I predictions, but no Class II predictions and two
            # Class I targets that could not be modelled faithfully.
            tag = f"med {v.median():.2f}" + (f" (n = {len(v)})" if len(v) != n_full else "")
            ax.text(i, 1.075, tag, ha="center", va="bottom", fontsize=11)
    for y, lab in [(0.23, "acceptable"), (0.49, "medium"), (0.80, "high")]:
        ax.axhline(y, ls="--", lw=1, color="gray", zorder=0)
        ax.text(-0.46, y + .012, lab, fontsize=8, color="gray", ha="left")
    ax.set_ylabel(ylabel); ax.set_xlabel(""); ax.set_title(title, loc="left", pad=26)
    ax.set_ylim(-0.02, 1.05); ax.set_xlim(-0.5, len(order) - 0.5)
    ax.tick_params(axis="x", rotation=20)
    for t in ax.get_xticklabels():
        t.set_ha("right")


# ======================= Figure 1A : interface accuracy =================
data, order = load()
fig, ax = plt.subplots(figsize=(COLW * len(order), 7.0))
n = data.Method.value_counts().max() if len(data) else 0
strip_box(ax, data, order, "Global DockQ",
          rf"$\bf{{A}}$  Interface accuracy" + (f" (n = {n})" if n else ""))
os.makedirs(OUT, exist_ok=True)
fig.savefig(f"{OUT}/Figure_1A.png", bbox_inches="tight")
plt.close(fig)
print(f"{OUT}/Figure_1A.png  ({len(data)} points, {len(order)} columns)")
