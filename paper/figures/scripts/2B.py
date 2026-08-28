"""Global DockQ (style.global_dockq) vs native resolution, one colour + one trend line per model.

Data: output/DockQ/dockq_all.csv (resolution/method come from natives_manifest.csv).
Cryo-EM entries with no deposited resolution ('.') drop out via to_numeric(coerce).
Gray-area entries are dropped here, at analysis time (CLAUDE.md).
"""
import glob, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/14TBDrive/6TBDrive1_backup/benchmark_fresh"
OUT = f"{BASE}/paper/figures/outputs"

MODELS = ["AF3", "Protenix", "ESMFold2"]
from style import COLORS, BASE_COLORS, keep, load_dockq, global_dockq
BANDS = [(0.00, 0.23, "#f7dfe0", "Incorrect", "#b5545c"),
         (0.23, 0.49, "#fdf1dd", "Acceptable", "#d38b2a"),
         (0.49, 0.80, "#e4f1fa", "Medium", "#3f86c4"),
         (0.80, 1.05, "#e7e2f3", "High", "#5a4a9c")]

plt.rcParams.update({"font.size": 13, "axes.labelsize": 14,
                     "xtick.labelsize": 11, "ytick.labelsize": 11,
                     "savefig.dpi": 200, "figure.dpi": 200})

d = global_dockq(keep(load_dockq()), ["resolution"])
d["resolution"] = pd.to_numeric(d.resolution, errors="coerce")
d = d.dropna(subset=["resolution"])
assert d.resolution.between(0.5, 10).all(), d.resolution.describe()

fig, ax = plt.subplots(figsize=(6.0, 5.6))
for lo, hi, fill, lab, tc in BANDS:
    ax.axhspan(lo, hi, color=fill, zorder=0)
    ax.text(0.985, (lo + min(hi, 1.0)) / 2, lab, transform=ax.get_yaxis_transform(),
            ha="right", va="center", fontsize=10.5, color=tc, fontweight="bold")

rtext, n = [], {}
for m in MODELS:
    sub = d[d.model == m]
    if not len(sub):
        continue
    n[m] = len(sub)
    c = COLORS[m]
    ax.scatter(sub.resolution, sub.global_dockq, s=38, facecolor=c, edgecolor="black",
               linewidth=.7, alpha=.9, zorder=3, label=f"{m} (n={len(sub)})")
    if len(sub) > 2:
        r = np.corrcoef(sub.resolution, sub.global_dockq)[0, 1]
        b, a = np.polyfit(sub.resolution, sub.global_dockq, 1)
        xs = np.linspace(sub.resolution.min(), sub.resolution.max(), 2)
        ax.plot(xs, a + b * xs, "--", color=BASE_COLORS[m], lw=2.5, zorder=4)
        rtext.append((m, r, BASE_COLORS[m]))

for i, (m, r, c) in enumerate(rtext):
    ax.text(.03, .045 + .062 * (len(rtext) - 1 - i), rf"$r$={r:.2f}", transform=ax.transAxes,
            fontsize=13, color=c, fontweight="bold", va="bottom")
if not n:
    ax.text(.5, .5, "no DockQ results yet", transform=ax.transAxes,
            ha="center", color="gray", fontsize=11)

ax.set_xlabel(u"Native resolution (Å)"); ax.set_ylabel(r"Global DockQ")
ax.set_ylim(-0.02, 1.02)
ax.legend(title="Model", loc="upper center", bbox_to_anchor=(.5, -.14),
          ncol=min(3, max(1, len(n))), frameon=True, fontsize=11, title_fontsize=12)

os.makedirs(OUT, exist_ok=True)
fig.savefig(f"{OUT}/Figure_2B.png", bbox_inches="tight")
plt.close(fig)
print(f"{OUT}/Figure_2B.png  " + ", ".join(f"{m}:{v}" for m, v in n.items()))
