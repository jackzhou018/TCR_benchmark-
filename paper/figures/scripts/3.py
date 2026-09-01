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
fig, ax = plt.subplots(1, 3, figsize=(12.4, 3.8))

# A -- divergence vs the worse of the two predictions
a = ax[0]
a.scatter(D.mean_rmsd, D.minq, s=26, c="#7a7a7a", alpha=.65, edgecolors="none")
rho = stats.spearmanr(D.mean_rmsd, D.minq)
a.set_xscale("log")
a.axhline(0.49, ls="--", lw=1, c="#777777")
a.set_xlabel("AF3 vs ESMFold2 C$\\alpha$ RMSD (\\AA)" if False else "AF3 vs ESMFold2 Cα RMSD (Å)")
a.set_ylabel("min(AF3, ESMFold2) Global DockQ")
a.set_title(r"$\bf{A}$  Cross-model divergence vs accuracy", loc="left")
a.text(.04, .07, f"Spearman ρ = {rho[0]:.3f}\np = {rho[1]:.1e}", transform=a.transAxes,
       va="bottom", ha="left", fontsize=9)
a.text(.97, .52, "medium", transform=a.transAxes, ha="right", fontsize=8, color="#777777")

# B -- what the two models actually agree on, by anatomy
b = ax[1]
PARTS = [("Whole complex", "mean_rmsd"),
         ("MHC-TCRβ interface", "MHC_TCRb"),
         ("MHC-TCRα interface", "MHC_TCRa"),
         ("TCR fold (internal)", "tcr_internal_rmsd"),
         ("Peptide in MHC frame", "peptide_in_MHC_frame"),
         ("Peptide conformation", "peptide_local")]
BINS = [0, 1, 2, 5, np.inf]
BLAB = ["< 1 Å", "1–2 Å", "2–5 Å", "≥ 5 Å"]
BCOL = ["#dcdcdc", "#a8b6c4", "#5f7d99", "#2f4257"]   # ordinal ramp: agree -> disagree
for i, (name, col) in enumerate(PARTS):
    x = D[col].dropna()
    pct = pd.cut(x, BINS, labels=BLAB).value_counts(normalize=True).reindex(BLAB) * 100
    left = 0.0
    for lab, c in zip(BLAB, BCOL):
        v = pct[lab]
        b.barh(i, v, left=left, color=c, height=.68,
               label=lab if i == 0 else None, edgecolor="white", lw=.8)
        if v >= 7:
            b.text(left + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=8.5,
                   color="white" if c in BCOL[2:] else "#333333")
        left += v
    b.text(101, i, f"med {x.median():.2f} Å", va="center", fontsize=8, color="#555555")
b.set_yticks(range(len(PARTS))); b.set_yticklabels([p[0] for p in PARTS])
b.set_xlim(0, 124); b.set_xticks([0, 25, 50, 75, 100])
b.set_xlabel("% of 126 complexes")
b.set_title(r"$\bf{B}$  AF3 vs ESMFold2 agreement by region", loc="left")
b.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(.44, -.20),
         handlelength=1.1, columnspacing=1.0, fontsize=8.5)
for s in ("top", "right", "left"):
    b.spines[s].set_visible(False)
b.tick_params(axis="y", length=0)

# C -- complementarity at the medium-quality cut
c = ax[2]
ok_a, ok_e = D.af3_dockq >= .49, D.esm_dockq >= .49
counts = [int((ok_a & ok_e).sum()), int((ok_a & ~ok_e).sum()),
          int((~ok_a & ok_e).sum()), int((~ok_a & ~ok_e).sum())]
lab = ["both", "AF3\nonly", "ESMFold2\nonly", "neither"]
# both/neither are not models: keep them neutral, or they read as Protenix and AF3
cols = ["#3f3f3f", BASE_COLORS["AF3"], BASE_COLORS["ESMFold2"], "#b0b0b0"]
c.bar(lab, counts, color=cols, alpha=.75)
for i, v in enumerate(counts):
    c.text(i, v + 1.5, str(v), ha="center", fontsize=10)
c.set_ylabel("complexes (of 126)"); c.set_ylim(0, 118)
c.set_title(r"$\bf{C}$  Medium-or-better predictions", loc="left")
c.text(.5, .78, f"union: {counts[0]+counts[1]+counts[2]}/126", transform=c.transAxes,
       ha="center", fontsize=9)

fig.tight_layout()
fig.savefig(f"{OUT}/Figure_divergence.png", dpi=300)
print("wrote", f"{OUT}/Figure_divergence.png", counts, rho)
