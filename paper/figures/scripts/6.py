"""Figure 6: what accuracy tracks besides the model -- closeness to the pre-cutoff record,
and the quality of the reference it is scored against.

Data: analysis/leakage_audit/results/audit_final.csv (the six-test audit of the 126 targets
against 259 TCR-pMHC complexes released on or before 2021-09-30) joined to the same
Global DockQ every other panel plots. Resolution comes from the DockQ tables, as in 2B.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from style import COLORS, BASE_COLORS, keep, load_dockq, global_dockq

BASE = "/14TBDrive/6TBDrive1_backup/benchmark_fresh"
OUT = f"{BASE}/paper/figures/outputs"
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5})
BANDS = [(0.00, 0.23, "#f7dfe0"), (0.23, 0.49, "#fdf1dd"),
         (0.49, 0.80, "#e4f1fa"), (0.80, 1.02, "#e7e2f3")]
MODELS = ["AF3", "Protenix", "ESMFold2", "TCRmodel2"]

A = pd.read_csv(f"{BASE}/analysis/leakage_audit/results/audit_final.csv")
d = global_dockq(keep(load_dockq()), ["resolution"]).merge(A, left_on="pdb_id", right_on="target")
d["resolution"] = pd.to_numeric(d.resolution, errors="coerce")
d["klass"] = np.where(d.rule3_strong, "near-duplicate",
             np.where(d.rule2_violation, "TCR-redundant",
             np.where(d.rule3_violation | d.tm_violation, "marginal", "clean")))

fig, ax = plt.subplots(2, 3, figsize=(11.5, 7.0))
def bands(a):
    for lo, hi, c in BANDS:
        a.axhspan(lo, hi, color=c, zorder=0)

def scatter_panel(a, col, xlab, letter, title, logx=False):
    bands(a)
    txt = []
    for m in MODELS:
        g = d[d.model == m].dropna(subset=[col])
        if len(g) < 10:
            continue
        a.scatter(g[col], g.global_dockq, s=13, facecolor=COLORS[m], edgecolor="black",
                  linewidth=.3, alpha=.8, zorder=3, label=f"{m} (n={len(g)})")
        b1, b0 = np.polyfit(g[col], g.global_dockq, 1)
        xs = np.linspace(g[col].min(), g[col].max(), 2)
        a.plot(xs, b0 + b1 * xs, "--", color=BASE_COLORS[m], lw=1.8, zorder=4)
        txt.append((m, stats.spearmanr(g[col], g.global_dockq)))
    for i, (m, r) in enumerate(txt):
        a.text(.025, .045 + .058 * (len(txt) - 1 - i),
               f"ρ={r.statistic:+.2f}" + ("*" if r.pvalue < .05 else ""),
               transform=a.transAxes, fontsize=8.5, color=BASE_COLORS[m],
               fontweight="bold", va="bottom")
    a.set_xlabel(xlab); a.set_ylabel("Global DockQ"); a.set_ylim(-0.02, 1.02)
    a.set_title(rf"$\bf{{{letter}}}$  {title}", loc="left")

scatter_panel(ax[0, 0], "max_tcr_concat", "concatenated TCR Vα+Vβ identity (%)",
              "A", "TCR familiarity")
scatter_panel(ax[0, 1], "closest_mean_id", "mean identity, closest complex (%)",
              "B", "Whole-complex familiarity")
scatter_panel(ax[0, 2], "tm_full", "multimer TM-score to closest pre-cutoff complex",
              "C", "Structural familiarity")
ax[0, 0].legend(loc="upper left", frameon=True, framealpha=.9)
scatter_panel(ax[1, 0], "resolution", "reference resolution (Å)", "D", "Reference quality")

# E -- accuracy by leakage class
ORDER = ["clean", "marginal", "TCR-redundant", "near-duplicate"]
LAB = ["clean\nof every rule", "marginal", "TCR-\nredundant", "near-\nduplicate"]
e = ax[1, 1]; bands(e)
w = 0.2
for i, m in enumerate(MODELS):
    data = [d[(d.model == m) & (d.klass == k)].global_dockq.values for k in ORDER]
    pos = [j + (i - 1.5) * w for j in range(len(ORDER))]
    bp = e.boxplot(data, positions=pos, widths=w * .82, patch_artist=True,
                   showfliers=False, medianprops=dict(color="black", lw=1), zorder=3)
    for p in bp["boxes"]:
        p.set_facecolor(BASE_COLORS[m]); p.set_alpha(.6); p.set_edgecolor(BASE_COLORS[m])
e.set_xticks(range(len(ORDER))); e.set_xticklabels(LAB)
e.set_xlim(-.5, len(ORDER) - .5); e.set_ylim(-0.02, 1.02); e.set_ylabel("Global DockQ")
n = {k: (d[(d.model == "AF3") & (d.klass == k)].shape[0]) for k in ORDER}
e.set_xlabel("  ".join(f"n={n[k]}" for k in ORDER), fontsize=8.5)
e.set_title(r"$\bf{E}$  Accuracy by leakage class", loc="left")

# F -- near-duplicate minus clean, per method
f = ax[1, 2]
rng = np.random.default_rng(0)
for i, m in enumerate(MODELS):
    g = d[d.model == m]
    x = g[g.klass == "near-duplicate"].global_dockq.values
    y = g[g.klass == "clean"].global_dockq.values
    delta = np.median(x) - np.median(y)
    bs = np.array([np.median(rng.choice(x, len(x))) - np.median(rng.choice(y, len(y)))
                   for _ in range(10000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    f.errorbar(delta, i, xerr=[[delta - lo], [hi - delta]], fmt="o", color=BASE_COLORS[m],
               ms=6, lw=1.8, capsize=3.5, zorder=3)
    f.text(hi + .012, i, f"{delta:+.2f}", va="center", fontsize=8.5,
           color=BASE_COLORS[m], fontweight="bold")
f.axvline(0, color="black", lw=1)
f.set_yticks(range(len(MODELS))); f.set_yticklabels(MODELS)
f.set_ylim(-.6, len(MODELS) - .4); f.set_xlim(-.08, .45)
f.set_xlabel("median Global DockQ,  near-duplicate − clean")
f.set_title(r"$\bf{F}$  What each method gains", loc="left")

fig.tight_layout()
fig.savefig(f"{OUT}/Figure_leakage.png", dpi=300)
print("wrote Figure_leakage.png")
