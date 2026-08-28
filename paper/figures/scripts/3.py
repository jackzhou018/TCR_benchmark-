"""Figure 3: where AF3 and ESMFold2 disagree, and what that disagreement predicts.

Data: RMSD_diff/divergence_with_metadata.csv (Ca RMSD between the two predictions of the
same FASTA, plus the rigid-body decomposition of the TCR disagreement).
"""
import os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from style import BASE_COLORS

BASE = "/14TBDrive/6TBDrive1_backup/benchmark_fresh"
OUT = f"{BASE}/paper/figures/outputs"
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9})

D = pd.read_csv(f"{BASE}/RMSD_diff/divergence_with_metadata.csv")
D["minq"] = D[["af3_dockq", "esm_dockq"]].min(axis=1)
fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.5))

# A -- divergence vs the worse of the two predictions
a = ax[0]
a.scatter(D.mean_rmsd, D.minq, s=26, c="#7a7a7a", alpha=.65, edgecolors="none")
rho = stats.spearmanr(D.mean_rmsd, D.minq)
a.set_xscale("log")
a.axhline(0.49, ls="--", lw=1, c="#b5545c")
a.set_xlabel("AF3 vs ESMFold2 C$\\alpha$ RMSD (\\AA)" if False else "AF3 vs ESMFold2 Cα RMSD (Å)")
a.set_ylabel("min(AF3, ESMFold2) Global DockQ")
a.set_title("A  Cross-model divergence vs accuracy")
a.text(.04, .07, f"Spearman ρ = {rho[0]:.3f}\np = {rho[1]:.1e}", transform=a.transAxes,
       va="bottom", ha="left", fontsize=9)
a.text(.97, .52, "medium", transform=a.transAxes, ha="right", fontsize=8, color="#b5545c")

# B -- accuracy by divergence quartile
b = ax[1]
D["q"] = pd.qcut(D.mean_rmsd, 4, labels=["Q1\n(closest)", "Q2", "Q3", "Q4\n(most divergent)"])
w = 0.34
for i, (m, col) in enumerate([("af3_dockq", BASE_COLORS["AF3"]), ("esm_dockq", BASE_COLORS["ESMFold2"])]):
    data = [D.loc[D.q == q, m].values for q in D.q.cat.categories]
    pos = np.arange(4) + (i - .5) * w
    bp = b.boxplot(data, positions=pos, widths=w * .88, patch_artist=True, showfliers=False,
                   medianprops=dict(color="black", lw=1.2))
    for patch in bp["boxes"]:
        patch.set_facecolor(col); patch.set_alpha(.55); patch.set_edgecolor(col)
b.set_xticks(range(4)); b.set_xticklabels(D.q.cat.categories)
b.set_ylabel("Global DockQ"); b.set_ylim(0, 1)
b.axhline(0.49, ls="--", lw=1, c="#b5545c")
b.set_title("B  Accuracy by divergence quartile")
b.legend(handles=[plt.Line2D([], [], color=BASE_COLORS[k], lw=6, alpha=.55, label=k)
                  for k in ("AF3", "ESMFold2")], loc="lower left", frameon=False)

# C -- complementarity at the medium-quality cut
c = ax[2]
ok_a, ok_e = D.af3_dockq >= .49, D.esm_dockq >= .49
counts = [int((ok_a & ok_e).sum()), int((ok_a & ~ok_e).sum()),
          int((~ok_a & ok_e).sum()), int((~ok_a & ~ok_e).sum())]
lab = ["both", "AF3\nonly", "ESMFold2\nonly", "neither"]
cols = ["#5a4a9c", BASE_COLORS["AF3"], BASE_COLORS["ESMFold2"], "#b5545c"]
c.bar(lab, counts, color=cols, alpha=.75)
for i, v in enumerate(counts):
    c.text(i, v + 1.5, str(v), ha="center", fontsize=10)
c.set_ylabel("complexes (of 126)"); c.set_ylim(0, 118)
c.set_title("C  Medium-or-better predictions (DockQ ≥ 0.49)")
c.text(.5, .78, f"union: {counts[0]+counts[1]+counts[2]}/126", transform=c.transAxes,
       ha="center", fontsize=9)

fig.tight_layout()
fig.savefig(f"{OUT}/Figure_divergence.png", dpi=300)
print("wrote", f"{OUT}/Figure_divergence.png", counts, rho)
