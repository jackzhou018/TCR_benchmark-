"""Two figures: identity to the pre-cutoff record vs Global DockQ, and accuracy by audit class."""
import sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
sys.path.insert(0, "/14TBDrive/6TBDrive1_backup/benchmark_fresh/paper/figures/scripts")
from style import COLORS, BASE_COLORS

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5})
BANDS = [(0.00, 0.23, "#f7dfe0"), (0.23, 0.49, "#fdf1dd"),
         (0.49, 0.80, "#e4f1fa"), (0.80, 1.02, "#e7e2f3")]
MODELS = ["AF3", "Protenix", "ESMFold2", "TCRmodel2"]
D = pd.read_csv("dockq_with_audit.csv")

def bands(a):
    for lo, hi, c in BANDS:
        a.axhspan(lo, hi, color=c, zorder=0)

# ================= Figure A: identity vs accuracy =================
PAN = [("max_tcr_concat", "concatenated TCR Vα+Vβ identity to closest pre-cutoff complex (%)"),
       ("max_tcrb", "TCR Vβ identity (%)"),
       ("closest_mean_id", "mean identity across all chains, closest complex (%)"),
       ("tm_full", "Foldseek multimer TM-score, native vs closest pre-cutoff complex")]
fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))
for k, (col, xlab) in enumerate(PAN):
    a = axes.flat[k]
    bands(a)
    txt = []
    for m in MODELS:
        g = D[(D.model == m)].dropna(subset=[col])
        if len(g) < 10: continue
        a.scatter(g[col], g.global_dockq, s=17, facecolor=COLORS[m], edgecolor="black",
                  linewidth=.35, alpha=.85, zorder=3, label=f"{m} (n={len(g)})")
        b1, b0 = np.polyfit(g[col], g.global_dockq, 1)
        xs = np.linspace(g[col].min(), g[col].max(), 2)
        a.plot(xs, b0 + b1 * xs, "--", color=BASE_COLORS[m], lw=2, zorder=4)
        rho = stats.spearmanr(g[col], g.global_dockq)
        txt.append((m, rho.statistic, rho.pvalue))
    for i, (m, r, p) in enumerate(txt):
        a.text(.025, .085 + .058 * (len(txt) - 1 - i), f"ρ={r:+.2f}" + ("*" if p < .05 else ""),
               transform=a.transAxes, fontsize=9, color=BASE_COLORS[m], fontweight="bold", va="bottom")
    a.set_xlabel(xlab); a.set_ylabel("Global DockQ"); a.set_ylim(-0.02, 1.02)
    a.set_title(rf"$\bf{{{'ABCD'[k]}}}$", loc="left", fontsize=14)
    if k == 0:
        a.legend(loc="upper left", bbox_to_anchor=(0, 1.0), frameon=True, framealpha=.9)
fig.suptitle("Similarity to the pre-cutoff structural record vs prediction accuracy  ·  126 complexes",
             y=.985, fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, .965])
fig.savefig("fig_identity_vs_dockq.png", dpi=200)
print("wrote fig_identity_vs_dockq.png")

# ================= Figure B: accuracy by audit class =================
ORDER = ["clean (n=42)", "marginal (n=42)", "TCR-redundant (n=32)", "near-duplicate (n=10)"]
SHORT = ["clean\nof every rule", "marginal\n(>40% or TM>0.9)", "TCR-redundant\n(90/95% rule)",
         "near-duplicate\n(≥70% all chains)"]
fig, ax = plt.subplots(1, 2, figsize=(11.0, 3.9), gridspec_kw={"width_ratios": [1.55, 1]})

a = ax[0]; bands(a)
w = 0.2
for i, m in enumerate(MODELS):
    data, pos = [], []
    for j, k in enumerate(ORDER):
        v = D[(D.model == m) & (D.klass == k)].global_dockq.values
        data.append(v); pos.append(j + (i - 1.5) * w)
    bp = a.boxplot(data, positions=pos, widths=w * .82, patch_artist=True, showfliers=False,
                   medianprops=dict(color="black", lw=1.1), zorder=3)
    for patch in bp["boxes"]:
        patch.set_facecolor(BASE_COLORS[m]); patch.set_alpha(.6); patch.set_edgecolor(BASE_COLORS[m])
a.set_xticks(range(len(ORDER))); a.set_xticklabels(SHORT)
a.set_xlim(-.5, len(ORDER) - .5); a.set_ylim(-0.02, 1.02)
a.set_ylabel("Global DockQ")
a.legend(handles=[plt.Line2D([], [], color=BASE_COLORS[m], lw=6, alpha=.6, label=m) for m in MODELS],
         loc="lower right", ncol=2, frameon=True, framealpha=.9)
a.set_title(r"$\bf{A}$  Accuracy by leakage class", loc="left")

b = ax[1]
rng = np.random.default_rng(0)
for i, m in enumerate(MODELS):
    g = D[D.model == m]
    x = g[g.klass.str.startswith("near")].global_dockq.values
    y = g[g.klass.str.startswith("clean")].global_dockq.values
    d = np.median(x) - np.median(y)
    bs = np.array([np.median(rng.choice(x, len(x))) - np.median(rng.choice(y, len(y))) for _ in range(10000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    b.errorbar(d, i, xerr=[[d - lo], [hi - d]], fmt="o", color=BASE_COLORS[m],
               ms=7, lw=2, capsize=4, zorder=3)
    b.text(hi + .012, i, f"{d:+.2f}", va="center", fontsize=9, color=BASE_COLORS[m], fontweight="bold")
b.axvline(0, color="black", lw=1)
b.set_yticks(range(len(MODELS))); b.set_yticklabels(MODELS); b.set_ylim(-.6, len(MODELS) - .4)
b.set_xlabel("median Global DockQ,  near-duplicate − clean")
b.set_xlim(-.07, .42)
b.set_title(r"$\bf{B}$  Lift on the 10 near-duplicates", loc="left")
fig.tight_layout()
fig.savefig("fig_dockq_by_class.png", dpi=200)
print("wrote fig_dockq_by_class.png")
