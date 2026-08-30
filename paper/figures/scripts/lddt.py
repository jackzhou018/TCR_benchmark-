"""Figure: superposition-free CDR accuracy (lDDT) against the models' own pLDDT.

Data: output/lDDT/cdr_lddt_long.csv (scripts/cdr_lddt.py) -- one row per
(model, config, pdb_id, region) with the complex-frame lDDT, the intra-chain lDDT and the
mean predicted lDDT over the same residues.

A  all-CDR lDDT in both frames: with the whole complex as contact partners (docking included)
   and with only the CDR's own TCR chain (loop conformation alone).
B  CDR3 calibration: predicted lDDT against the lDDT actually achieved, complex frame.
C  the same overestimate by region, framework through CDR3.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from style import BASE_COLORS, keep

BASE = "/14TBDrive/6TBDrive1_backup/benchmark_fresh"
OUT = f"{BASE}/paper/figures/outputs"
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9})

MODELS = ["AF3", "Protenix", "ESMFold2", "TCRmodel2"]
# one config per model -- the same top-ranked prediction run_dockq.py scores
CONFIG = {"TCRmodel2": "cutoff_20210930_chains"}

D = pd.read_csv(f"{BASE}/output/lDDT/cdr_lddt_long.csv")
D = keep(D[D.status == "scored"])
D = D[[CONFIG.get(m, c) == c for m, c in zip(D.model, D.config)]]
D = D[D.complete == 1]
D["plddt"] = D.plddt / 100.0                      # onto lDDT's 0-1 scale


def region(name, col="lddt"):
    return D[D.region == name].set_index(["model", "pdb_id"])[col]


fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.5))

# A -- all-CDR lDDT, whole-complex frame vs the CDR's own chain
a = ax[0]
w = 0.34
for i, (col, lab, alpha) in enumerate([("lddt", "vs whole complex", .55),
                                       ("lddt_intra", "vs own TCR chain", .22)]):
    data = [region("allCDR", col).loc[m].dropna().values for m in MODELS]
    bp = a.boxplot(data, positions=np.arange(len(MODELS)) + (i - .5) * w, widths=w * .88,
                   patch_artist=True, showfliers=False, medianprops=dict(color="black", lw=1.2))
    for patch, m in zip(bp["boxes"], MODELS):
        patch.set_facecolor(BASE_COLORS[m]); patch.set_alpha(alpha)
        patch.set_edgecolor(BASE_COLORS[m])
a.set_xticks(range(len(MODELS))); a.set_xticklabels(MODELS, fontsize=8.5)
a.set_ylabel("all-CDR lDDT"); a.set_ylim(0, 1)
a.set_title(r"$\bf{A}$  CDR lDDT: docked vs local", loc="left")
a.legend(handles=[plt.Line2D([], [], color="#666666", lw=6, alpha=al, label=l)
                  for al, l in ((.55, "vs whole complex"), (.22, "vs own TCR chain"))],
         loc="lower left", frameon=False)

# B -- does pLDDT know? CDR3 (both chains pooled), complex frame
b = ax[1]
cdr3 = pd.concat([pd.concat([region(f"CDR3{t}"), region(f"CDR3{t}", "plddt")], axis=1)
                  for t in "ab"]).dropna()
for m in MODELS:
    d = cdr3.loc[m]
    b.scatter(d.plddt, d.lddt, s=14, color=BASE_COLORS[m], alpha=.35, edgecolors="none")
    r = stats.pearsonr(d.plddt, d.lddt)[0]
    x = np.linspace(*np.percentile(d.plddt, [2, 98]), 2)
    b.plot(x, np.polyval(np.polyfit(d.plddt, d.lddt, 1), x), color=BASE_COLORS[m], lw=1.6,
           label=f"{m}  r = {r:.2f}")
b.plot([0.4, 1], [0.4, 1], ls="--", lw=1, c="#999999", label="pLDDT = lDDT")
b.set_xlabel("mean CDR3 pLDDT"); b.set_ylabel("CDR3 lDDT (whole complex)")
b.set_xlim(0.4, 1.0); b.set_ylim(0, 1)
b.set_title(r"$\bf{B}$  Predicted vs achieved, CDR3", loc="left")
b.legend(loc="lower right", frameon=False, fontsize=8)

# C -- the overestimate, region by region
c = ax[2]
REGIONS = [("FR", "framework"), ("CDR1", "CDR1"), ("CDR2", "CDR2"),
           ("CDR2.5", "CDR2.5"), ("CDR3", "CDR3")]
for m in MODELS:
    y = [float((pd.concat([region(f"{r}{t}", "plddt") - region(f"{r}{t}") for t in "ab"])
                .loc[m].mean())) for r, _ in REGIONS]
    c.plot(range(len(REGIONS)), y, "o-", color=BASE_COLORS[m], lw=1.8, ms=5, label=m)
c.axhline(0, ls="--", lw=1, c="#999999")
c.set_xticks(range(len(REGIONS))); c.set_xticklabels([l for _, l in REGIONS])
c.set_ylabel("mean pLDDT − lDDT")
c.set_title(r"$\bf{C}$  Confidence overestimate by region", loc="left")
c.legend(loc="upper left", frameon=False, fontsize=8)

fig.tight_layout()
fig.savefig(f"{OUT}/Figure_lddt.png", dpi=300)
print("wrote", f"{OUT}/Figure_lddt.png")
