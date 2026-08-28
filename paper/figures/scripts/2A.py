"""Figure 2A: Global DockQ vs ipTM, one colour + one trend line per model.

Data: output/DockQ/dockq_all.csv (global_dockq, one row per interface -> collapsed per pdb)
joined on pdb_id to models/<MODEL>/results/<config>_results.csv (iptm).
Gray-area entries are dropped here, at analysis time (CLAUDE.md).
"""
import glob, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/14TBDrive/6TBDrive1_backup/benchmark_fresh"
OUT = f"{BASE}/paper/figures/outputs"

MODELS = ["AF3", "Protenix", "ESMFold2"]
XLIM = (0.0, 1.0)
from style import COLORS, BASE_COLORS, keep, load_dockq, global_dockq
BANDS = [(0.00, 0.23, "#f7dfe0", "Incorrect", "#b5545c"),
         (0.23, 0.49, "#fdf1dd", "Acceptable", "#d38b2a"),
         (0.49, 0.80, "#e4f1fa", "Medium", "#3f86c4"),
         (0.80, 1.05, "#e7e2f3", "High", "#5a4a9c")]

plt.rcParams.update({"font.size": 13, "axes.labelsize": 14,
                     "xtick.labelsize": 11, "ytick.labelsize": 11,
                     "savefig.dpi": 200, "figure.dpi": 200})


def load():
    """-> {model: DataFrame[pdb_id, iptm, dockq]}, using every config found per model."""
    dq = global_dockq(keep(load_dockq()))

    out = {}
    for m in MODELS:
        conf = []
        for c in sorted(glob.glob(f"{BASE}/models/{m}/results/*_results.csv")):
            cfg = os.path.basename(c).replace("_results.csv", "")
            r = pd.read_csv(c)
            if "iptm" not in r:
                continue
            sub = dq[(dq.model == m) & (dq.config == cfg)]
            conf.append(sub.merge(r[["pdb_id", "iptm"]], on="pdb_id"))
        if conf:
            d = pd.concat(conf).dropna(subset=["iptm", "global_dockq"])
            if len(d):
                out[m] = d
    return out


data = load()
fig, ax = plt.subplots(figsize=(6.0, 5.6))

for lo, hi, fill, lab, tc in BANDS:
    ax.axhspan(lo, hi, color=fill, zorder=0)
    # labels sit left of the data (no point falls below ipTM 0.4) -- the right edge is
    # covered once all three models are plotted.
    ax.text(0.36, (lo + min(hi, 1.0)) / 2, lab, transform=ax.get_yaxis_transform(),
            ha="right", va="center", fontsize=10.5, color=tc, fontweight="bold")

rtext = []
for m in MODELS:
    d = data.get(m)
    if d is None:
        continue
    c = COLORS[m]
    ax.scatter(d.iptm, d.global_dockq, s=38, facecolor=c, edgecolor="black",
               linewidth=.7, alpha=.9, zorder=3, label=f"{m} (n={len(d)})")
    if len(d) > 2:
        r = np.corrcoef(d.iptm, d.global_dockq)[0, 1]
        b, a = np.polyfit(d.iptm, d.global_dockq, 1)
        xs = np.array(XLIM)          # span the full axis, not just the data range
        # ponytail: this extrapolates the fit past the observed ipTM range (0.41-0.96)
        ax.plot(xs, a + b * xs, "--", color=BASE_COLORS[m], lw=2.5, zorder=4)
        rtext.append((m, r, BASE_COLORS[m]))

for i, (m, r, c) in enumerate(rtext):
    ax.text(.03, .955 - .062 * i, rf"$r$={r:.2f}", transform=ax.transAxes,
            fontsize=13, color=c, fontweight="bold", va="top")
if not data:
    ax.text(.5, .5, "no ipTM/DockQ results yet", transform=ax.transAxes,
            ha="center", color="gray", fontsize=11)

ax.set_xlabel("ipTM"); ax.set_ylabel(r"Global DockQ")
ax.set_xlim(*XLIM); ax.set_ylim(-0.02, 1.02)
ax.set_title(r"$\bf{A}$", loc="left", fontsize=17, pad=6)
ax.legend(title="Model", loc="upper center", bbox_to_anchor=(.5, -.14),
          ncol=min(3, max(1, len(data))), frameon=True, fontsize=11, title_fontsize=12)

os.makedirs(OUT, exist_ok=True)
fig.savefig(f"{OUT}/Figure_2A.png", bbox_inches="tight")
plt.close(fig)
print(f"{OUT}/Figure_2A.png  (" + ", ".join(f"{m}:{len(d)}" for m, d in data.items()) + ")")
