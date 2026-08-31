"""Apply the audit rules to the pairwise identity matrices."""
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/14TBDrive/6TBDrive1_backup/benchmark_fresh"); HERE = Path(__file__).parent
COV = 0.80
M = pd.read_csv(HERE / "pairwise_target_vs_ref.csv")
C = pd.read_csv(HERE / "tcr_concat_vs_ref.csv")
per = pd.read_csv(ROOT / "analysis/training_sequence_similarity/sequence_similarity_per_target.csv")
dates = pd.read_csv(HERE / "dates.csv")
X = M.merge(C, on=["target", "ref"])
cls = per.set_index("pdb_id").mhc_class

MHC_I = ["MHC_I_alpha1_alpha2"]; MHC_II = ["MHC_II_alpha1", "MHC_II_beta1"]
def mhc_cols(t): return MHC_I if cls[t] == "Class I" else MHC_II
def prot_cols(t): return mhc_cols(t) + ["TCR_alpha_variable", "TCR_beta_variable"]

rows = []
for t, g in X.groupby("target"):
    g = g.copy()
    mc = mhc_cols(t); pc = prot_cols(t)
    # --- coverage-gated identities
    for c in pc:
        g[c + "_ok"] = g[c].where(g[c + "_cov"] >= COV)
    g["tcr_concat_ok"] = g.tcr_concat.where(g.tcr_concat_cov >= COV)

    # --- rule 2: TCR redundancy
    r2_concat = g.loc[g.tcr_concat_ok.idxmax()] if g.tcr_concat_ok.notna().any() else None
    a_max = g.TCR_alpha_variable_ok.max(); b_max = g.TCR_beta_variable_ok.max()
    concat_max = g.tcr_concat_ok.max()
    hit2 = []
    if concat_max >= 90: hit2.append(("concat>=90", g.loc[g.tcr_concat_ok.idxmax(), "ref"], concat_max))
    if a_max >= 95: hit2.append(("Va>=95", g.loc[g.TCR_alpha_variable_ok.idxmax(), "ref"], a_max))
    if b_max >= 95: hit2.append(("Vb>=95", g.loc[g.TCR_beta_variable_ok.idxmax(), "ref"], b_max))

    # --- rule 3: every corresponding chain >40% against ONE reference
    ok3 = g[(g[[c + "_ok" for c in pc]] > 40).all(axis=1) & (g.peptide > 40)]
    all_chain_ref, all_chain_min = None, np.nan
    if len(ok3):
        mins = ok3[[c + "_ok" for c in pc] + ["peptide"]].min(axis=1)
        i = mins.idxmax(); all_chain_ref = ok3.loc[i, "ref"]; all_chain_min = mins.max()

    # --- closest complex overall (mean identity across corresponding chains)
    means = g[[c + "_ok" for c in pc] + ["peptide"]].mean(axis=1)
    j = means.idxmax(); closest = g.loc[j]
    rows.append({
        "target": t, "mhc_class": cls[t],
        "closest_ref": closest.ref, "closest_mean_id": means.max(),
        "tcra_id": closest.TCR_alpha_variable, "tcrb_id": closest.TCR_beta_variable,
        "pep_id": closest.peptide, "mhc_id": closest[mc].mean(),
        "tcr_concat_closest": closest.tcr_concat,
        "max_tcra": a_max, "max_tcrb": b_max, "max_tcr_concat": concat_max,
        "rule2_hits": "; ".join(f"{k}:{r}({v:.1f})" for k, r, v in hit2),
        "rule2_violation": bool(hit2),
        "rule3_ref": all_chain_ref, "rule3_min_id": all_chain_min,
        "rule3_violation": all_chain_ref is not None,
        # partial priors, evaluated over all references
        "pp_same_pmhc": bool(((g[mc].mean(axis=1) >= 90) & (g.peptide >= 90)).any()),
        "pp_same_mhc_other_pep": bool(((g[mc].mean(axis=1) >= 95) & (g.peptide < 50)).any()),
        "pp_similar_tcr_other_pmhc": bool(((g.tcr_concat_ok >= 80) & (g.peptide < 50)).any()),
        "best_mhc_id": g[mc].mean(axis=1).max(),
    })
R = pd.DataFrame(rows).merge(dates[["pdb_id", "deposit_date", "release_date_rcsb"]],
                             left_on="target", right_on="pdb_id").drop(columns="pdb_id")
R["temporal_violation"] = R.release_date_rcsb <= "2021-09-30"
R.to_csv(HERE / "rules.csv", index=False)
print(f"targets                          {len(R)}")
print(f"1. temporal violations           {int(R.temporal_violation.sum())}   "
      f"(deposited pre-cutoff: {int((R.deposit_date <= '2021-09-30').sum())})")
print(f"2. TCR 90/95 violations          {int(R.rule2_violation.sum())}")
print(f"3. all-chain >40% violations     {int(R.rule3_violation.sum())}")
print(f"   both 2 and 3                  {int((R.rule2_violation & R.rule3_violation).sum())}")
print(f"   either                        {int((R.rule2_violation | R.rule3_violation).sum())}")
clean = R[~(R.rule2_violation | R.rule3_violation)]
print(f"4. clean targets                 {len(clean)}   of which partial priors:")
print(f"     same/similar pMHC           {int(clean.pp_same_pmhc.sum())}")
print(f"     same MHC, other peptide     {int(clean.pp_same_mhc_other_pep.sum())}")
print(f"     similar TCR, other pMHC     {int(clean.pp_similar_tcr_other_pmhc.sum())}")
print(f"     any partial prior           {int((clean.pp_same_pmhc|clean.pp_same_mhc_other_pep|clean.pp_similar_tcr_other_pmhc).sum())}")
