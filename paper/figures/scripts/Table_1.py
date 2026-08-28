"""Table 1: TCR-specific structural accuracy, by model and MHC class.

Data: output/RMSD/tcr_cdr_rmsd_per_prediction.csv (scripts/tcr_cdr_rmsd.py), filtered with
style.keep() so the table covers the same complexes the figure panels do.

Renders a journal-style table (booktabs rules, one column group per model, footnotes defining
each metric) to Table_1.png, and writes the same numbers to Table_1.csv.
"""
import os
import textwrap

import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from style import MODELS, keep

BASE = "/14TBDrive/6TBDrive1_backup/benchmark_fresh"
OUT = f"{BASE}/paper/figures/outputs"
SRC = f"{BASE}/output/RMSD/tcr_cdr_rmsd_per_prediction.csv"

# order here is the row order of the rendered table
METRICS = [
    ("mhc_aligned_weighted_all_cdr_ca_rmsd",
     "MHC-aligned weighted all-CDR Cα RMSD",
     "Cα of all eight CDRs (CDR1, CDR2, CDR2.5 and CDR3 on TCRα and TCRβ) after a "
     "single rigid superposition of the predicted MHC peptide-binding platform onto the native, "
     "with CDR3 residues weighted 3 and all others 1. No second fit: this measures how "
     "accurately the TCR is docked on the pMHC."),
    ("framework_aligned_cdr3a_all_atom_rmsd",
     "Framework-aligned CDR3α all-atom RMSD",
     "All heavy atoms of the CDR3α loop after superimposing the predicted TCRα "
     "variable-domain framework on the native one. Independent of docking: this measures how "
     "accurately the loop itself is modelled."),
    ("framework_aligned_cdr3b_all_atom_rmsd",
     "Framework-aligned CDR3β all-atom RMSD",
     "As above for the CDR3β loop, using an independent fit of the TCRβ framework."),
]
CLASSES = ["Class I + II", "Class I", "Class II"]


def stats(v):
    q1, q3 = np.percentile(v, [25, 75])
    return dict(n=len(v), median=float(np.median(v)), q1=float(q1), q3=float(q3),
                mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else np.nan,
                lo=float(v.min()), hi=float(v.max()))


d = keep(pd.read_csv(SRC).query("status == 'scored'"))
# style.MODELS first, then anything else present (TCRmodel2 is not in the panel palette yet)
models = [m for m in MODELS if (d.model == m).any()] + \
         [m for m in d.model.unique() if m not in MODELS]

cells, rows = {}, []
for col, name, footnote in METRICS:
    for model in models:
        sub = d[d.model == model].dropna(subset=[col])
        for cls in CLASSES:
            v = (sub if cls == "Class I + II" else sub[sub.mhc_class == cls])[col].to_numpy(float)
            if not len(v):
                continue
            s = stats(v)
            cells[(col, cls, model)] = s
            rows.append(dict(metric=name, mhc_class=cls, model=model, units="angstrom",
                             n_structures=s["n"], median_A=round(s["median"], 2),
                             IQR_A=f"{s['q1']:.2f}-{s['q3']:.2f}",
                             mean_A=round(s["mean"], 2),
                             sd_A="" if np.isnan(s["sd"]) else round(s["sd"], 2),
                             min_A=round(s["lo"], 2), max_A=round(s["hi"], 2),
                             source_column=col, what_it_measures=footnote))

os.makedirs(OUT, exist_ok=True)
pd.DataFrame(rows).to_csv(f"{OUT}/Table_1.csv", index=False)

# ---------------------------------------------------------------- render
#
# Layout follows the conventions of the TCR-pMHC benchmarking literature: booktabs rules
# only (no verticals, no shading, no colour), quartile-based reporting (Bradley 2023 reports
# CDR RMSD distributions as quartiles), and the docking-accuracy / loop-accuracy split that
# Shi et al. (JCIM 2025) draw between framework and CDR3 performance.
#
# Two blocks, one per question the table answers. n depends only on model x class, not on
# metric, so it is stated once under each model header instead of twelve times in the body --
# that keeps the median RMSD values the most prominent thing on the page.

QUESTIONS = [
    ("Are the CDR loops positioned correctly relative to the pMHC?",
     ["mhc_aligned_weighted_all_cdr_ca_rmsd"]),
    ("Are the CDR3 loops themselves modelled accurately, independent of TCR docking?",
     ["framework_aligned_cdr3a_all_atom_rmsd", "framework_aligned_cdr3b_all_atom_rmsd"]),
]
SHORT = {"mhc_aligned_weighted_all_cdr_ca_rmsd": "MHC-aligned weighted all-CDR C\u03b1",
         "framework_aligned_cdr3a_all_atom_rmsd": "Framework-aligned CDR3\u03b1, all-atom",
         "framework_aligned_cdr3b_all_atom_rmsd": "Framework-aligned CDR3\u03b2, all-atom"}

GREY, MID = "#6b6b6b", "#8a8a8a"
FIG_W = 12.6
LABEL_W = 0.305
GROUP_W = (1.0 - LABEL_W) / len(models)
X0 = {m: LABEL_W + i * GROUP_W for i, m in enumerate(models)}
X_MED = {m: X0[m] + 0.070 for m in models}       # median, right-aligned: the decimal column
X_IQR = {m: X0[m] + 0.077 for m in models}       # [Q1-Q3], left-aligned, subordinated

FOOT = [
    "MHC-aligned: one rigid superposition of the predicted MHC peptide-binding platform onto "
    "the native, applied to the whole complex, with no further fit \u2014 deviation therefore "
    "reflects TCR docking. C\u03b1 atoms of all eight CDRs (CDR1, CDR2, CDR2.5, CDR3 on TCR\u03b1 "
    "and TCR\u03b2), CDR3 residues weighted 3, all others 1.",
    "Framework-aligned: each TCR chain is fitted on its own variable-domain framework, so its "
    "CDR3 deviation is independent of where the TCR docks. All heavy atoms of the loop.",
    "n is the number of complexes scored, given as Class I + II / Class I / Class II. "
    "Predictions are the same top-ranked model scored by the DockQ workflow. TCRmodel2 is a "
    "partial run (class I only, still in progress).",
]
FOOT = [textwrap.wrap(t, 132) for t in FOOT]

ROW_IN, FOOT_IN = 0.265, 0.20
n_body = sum(1.9 + len(cols) * (1 + len(CLASSES)) for _q, cols in QUESTIONS)
n_foot = sum(len(b) for b in FOOT) + 0.5 * len(FOOT)
fig_h = 1.75 + ROW_IN * n_body + FOOT_IN * n_foot

fig, ax = plt.subplots(figsize=(FIG_W, fig_h))
ax.set_axis_off()
ax.set_xlim(0, 1)
DY, LINE = ROW_IN / fig_h, FOOT_IN / fig_h


def rule(yy, x0=0.0, x1=1.0, lw=0.8, color="black"):
    ax.plot([x0, x1], [yy, yy], color=color, lw=lw, clip_on=False, solid_capstyle="butt")


y = 0.975
ax.text(0, y, "$\\bf{Table\\ 1.}$  Accuracy of CDR loop placement and CDR3 loop conformation "
              "in predicted TCR\u2013pMHC complexes.", fontsize=12.5, va="baseline")

y -= 0.95 * DY
rule(y, lw=1.6)                                                     # toprule
y -= 0.90 * DY
ax.text(LABEL_W + (1 - LABEL_W) / 2, y, "RMSD (\u00c5)", fontsize=12, fontweight="bold",
        ha="center", va="baseline")
ax.text(0, y, "lower is better", fontsize=9.3, style="italic", color=GREY, va="baseline")
rule(y - 0.32 * DY, LABEL_W + 0.004, 1.0, 0.7)                      # cmidrule over the models
y -= 1.02 * DY
for m in models:
    ax.text(X0[m] + GROUP_W / 2 - 0.006, y, m, fontsize=11, fontweight="bold",
            ha="center", va="baseline")
y -= 0.78 * DY
ax.text(0, y, "median [IQR]", fontsize=9.3, style="italic", color=GREY, va="baseline")
for m in models:
    ns = [cells.get((QUESTIONS[0][1][0], c, m), {}).get("n") for c in CLASSES]
    ax.text(X0[m] + GROUP_W / 2 - 0.006, y,
            "n = " + " / ".join("\u2013" if v is None else str(v) for v in ns),
            fontsize=8.6, color=MID, ha="center", va="baseline")
y -= 0.40 * DY
rule(y, lw=1.0)                                                     # midrule

for qi, (question, cols) in enumerate(QUESTIONS):
    y -= 1.05 * DY
    ax.text(0, y, question, fontsize=9.6, style="italic", color=GREY, va="center")
    for ci, col in enumerate(cols):
        y -= 1.02 * DY
        ax.text(0.012, y, SHORT[col], fontsize=10.6, fontweight="bold", va="center")
        y -= 0.92 * DY
        for cls in CLASSES:
            ax.text(0.032, y, cls, fontsize=9.8, va="center", color="#222222")
            for m in models:
                s = cells.get((col, cls, m))
                if s is None:
                    ax.text(X_MED[m] + 0.030, y, "\u2013", fontsize=10, ha="center",
                            va="center", color=MID)
                    continue
                ax.text(X_MED[m], y, f"{s['median']:.2f}", fontsize=11, ha="right", va="center")
                ax.text(X_IQR[m], y, f"[{s['q1']:.2f}\u2013{s['q3']:.2f}]", fontsize=8.9,
                        va="center", color=GREY)
            y -= 0.92 * DY
        if ci < len(cols) - 1:
            rule(y + 0.40 * DY, LABEL_W * 0.04, 1.0, 0.35, "#bbbbbb")   # hairline, same block
    if qi < len(QUESTIONS) - 1:
        y -= 0.10 * DY
        rule(y + 0.34 * DY, lw=0.7)                                 # rule between questions
y += 0.35 * DY
rule(y, lw=1.6)                                                     # bottomrule

y -= 1.05 * DY
for block in FOOT:
    for line in block:
        ax.text(0, y, line, fontsize=8.6, va="center", color="#444444")
        y -= LINE
    y -= 0.5 * LINE

ax.set_ylim(y - 0.2 * DY, 1.0)
fig.savefig(f"{OUT}/Table_1.png", dpi=230, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"{OUT}/Table_1.png + Table_1.csv  (" +
      ", ".join(f"{m}:{d[d.model == m].pdb_id.nunique()}" for m in models) + ")")
