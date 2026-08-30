"""Figure 4: the single-prediction composite score (MOGGER) against the published alternatives.

Data: mogger/results/{predictions,correlations}.csv -- one AF3 prediction
per structure, its own confidence outputs, no native and no second predictor.
Formatting follows 3.py/4.py: one 1x3 figure, bold left-aligned panel letters.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

BASE = "/14TBDrive/6TBDrive1_backup/benchmark_fresh"
OUT = f"{BASE}/paper/figures/outputs"
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9})

BANDS = [(0.00, 0.23, "#f7dfe0", "Incorrect", "#b5545c"),
         (0.23, 0.49, "#fdf1dd", "Acceptable", "#d38b2a"),
         (0.49, 0.80, "#e4f1fa", "Medium", "#3f86c4"),
         (0.80, 1.02, "#e7e2f3", "High", "#5a4a9c")]
MOG, RF, GREY = "#5a4a9c", "#b5545c", "#7a7a7a"

P = pd.read_csv(f"{BASE}/mogger/results/predictions.csv")
C = pd.read_csv(f"{BASE}/mogger/results/correlations.csv")
assert len(P) == 126, len(P)

fig, ax = plt.subplots(1, 2, figsize=(7.7, 3.5))

# A -- the score itself against what it is trying to predict
a = ax[0]
for lo, hi, fill, lab, tc in BANDS:
    a.axhspan(lo, hi, color=fill, zorder=0)
    a.text(.985, (lo + min(hi, 1.0)) / 2, lab, transform=a.get_yaxis_transform(),
           ha="right", va="center", fontsize=8, color=tc, fontweight="bold")
a.scatter(P.mogger_score, P.global_dockq, s=26, facecolor=MOG, edgecolor="black",
          linewidth=.5, alpha=.85, zorder=3)
b1, b0 = np.polyfit(P.mogger_score, P.global_dockq, 1)
xs = np.linspace(P.mogger_score.min(), P.mogger_score.max(), 2)
a.plot(xs, b0 + b1 * xs, "--", color=MOG, lw=2, zorder=4)
r = np.corrcoef(P.mogger_score, P.global_dockq)[0, 1]
rho = stats.spearmanr(P.mogger_score, P.global_dockq).statistic
a.text(.03, .04, f"$r$ = {r:.3f}\nρ = {rho:.3f}", transform=a.transAxes, va="bottom",
       fontsize=9, color=MOG, fontweight="bold")
a.set_xlabel("MOGGER (one AF3 prediction)"); a.set_ylabel("Global DockQ")
a.set_ylim(-0.02, 1.02)
a.set_title(r"$\bf{A}$  Composite score vs accuracy (n = 126)", loc="left")

# B -- every score on the same 126 complexes, with its bootstrap interval
b = ax[1]
SCORES = ["MOGGER (single AF3)", "Weakest pair ipTM", "pDockQ", "RF expected tier", "AF3 ipTM"]
SHORT = ["MOGGER", "weakest\npair ipTM", "pDockQ", "RF\ntier", "AF3\nipTM"]
c126 = C[C.subset == "All 126 (primary)"].set_index("score").loc[SCORES]
cols = [MOG, GREY, GREY, RF, GREY]
x, w = np.arange(len(SCORES)), 0.38
for i, (stat, lo, hi, hatch) in enumerate([("pearson_r", "pearson_ci_low", "pearson_ci_high", ""),
                                           ("spearman_rho", "spearman_ci_low", "spearman_ci_high", "///")]):
    v = c126[stat].values
    err = np.vstack([v - c126[lo].values, c126[hi].values - v])
    b.bar(x + (i - .5) * w, v, width=w * .9, color=cols, alpha=.8, hatch=hatch,
          edgecolor="black", linewidth=.7, yerr=err, capsize=2.5,
          error_kw=dict(lw=1, ecolor="black"))
b.set_xticks(x); b.set_xticklabels(SHORT)
b.set_ylabel("correlation with Global DockQ"); b.set_ylim(0, 0.95)
b.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", hatch=h,
                                label=l) for h, l in [("", "Pearson $r$"), ("///", "Spearman ρ")]],
         loc="upper right", frameon=False, ncol=2)
b.set_title(r"$\bf{B}$  Scores of one AF3 prediction (n = 126)", loc="left")

fig.tight_layout()
fig.savefig(f"{OUT}/Figure_mogger.png", dpi=300)
print("wrote Figure_mogger.png", f"r={r:.3f} rho={rho:.3f}")
