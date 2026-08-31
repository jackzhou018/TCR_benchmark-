from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).parent
R = pd.read_csv(HERE / "audit_full.csv").merge(pd.read_csv(HERE / "binary_pmhc.csv"), on="target")

def prior(r):
    p = []
    if r.pp_same_pmhc: p.append("same/similar pMHC")
    if r.binary_pmhc_prior: p.append("binary pMHC, no ternary")
    if r.pp_similar_tcr_other_pmhc: p.append("similar TCR, other pMHC")
    if r.pp_same_mhc_other_pep: p.append("same MHC, other peptide")
    return "; ".join(p) if p else "none"
R["partial_prior"] = R.apply(prior, axis=1)

def rec(r):
    if r.temporal_violation: return "Exclude—temporal leakage"
    if r.rule3_strong and r.rule2_violation: return "Exclude—complete-complex similarity"
    if r.rule3_strong: return "Manual review required"
    if r.rule2_violation: return "Exclude—high TCR redundancy"
    if r.rule3_violation or r.tm_violation: return "Retain in standard benchmark; exclude from strict subset"
    if r.cluster_size > 1: return "Combine with another target as one cluster"
    if r.partial_prior != "none": return "Retain—temporally blind with partial prior"
    return "Retain—strongly blind"
R["recommendation"] = R.apply(rec, axis=1)
R.to_csv(HERE / "audit_final.csv", index=False)

T = pd.DataFrame({
    "Target": R.target, "Release date": R.release_date_rcsb, "Deposited": R.deposit_date,
    "Temporal violation": np.where(R.temporal_violation, "YES", "no"),
    "Closest pre-cutoff complex": R.closest_ref,
    "TCRa identity": R.max_tcra.round(1), "TCRb identity": R.max_tcrb.round(1),
    "Concat TCR identity": R.max_tcr_concat.round(1),
    "Peptide identity": R.pep_id.round(1), "MHC identity": R.best_mhc_id.round(1),
    "All chains >40%?": np.where(R.rule3_violation, R.rule3_ref.fillna("") + " (" + R.rule3_min_id.round(0).astype("Int64").astype(str) + "%)", "no"),
    "Multimer TM": R.tm_full.round(3), "TM ref": R.tm_full_ref,
    "Partial prior": R.partial_prior, "Cluster": R.cluster + " (n=" + R.cluster_size.astype(str) + ")",
    "Rules triggered": R.rules_triggered, "Recommendation": R.recommendation})
T.to_csv(HERE / "audit_table.csv", index=False)

print("=" * 78)
print(f"{'targets':52s} {len(R)}")
print(f"{'1  failing temporal cutoff (release <= 2021-09-30)':52s} {int(R.temporal_violation.sum())}")
print(f"{'   deposited on or before the cutoff':52s} {int((R.deposit_date <= '2021-09-30').sum())}")
print(f"{'2  violating the 90/95% TCR rule':52s} {int(R.rule2_violation.sum())}")
print(f"{'3  violating the all-chain >40% rule':52s} {int(R.rule3_violation.sum())}")
print(f"{'   of those, >=70% on every chain':52s} {int(R.rule3_strong.sum())}")
print(f"{'5  multimer TM > 0.90 (complete, correct mapping)':52s} {int(R.tm_violation.sum())}")
print(f"{'4  clean of 2/3/5, partial priors only':52s} "
      f"{int((~(R.rule2_violation|R.rule3_violation|R.tm_violation) & (R.partial_prior!='none')).sum())}")
print(f"{'   clean of 2/3/5 with no prior at all':52s} "
      f"{int((~(R.rule2_violation|R.rule3_violation|R.tm_violation) & (R.partial_prior=='none')).sum())}")
print(f"{'6  independent clusters after deduplication':52s} {R.cluster.nunique()}")
print("=" * 78)
print(R.recommendation.value_counts().to_string())
print()
print("partial-prior categories (all 126):")
for k, c in [("same/similar pMHC", R.pp_same_pmhc), ("binary pMHC, no ternary", R.binary_pmhc_prior),
             ("similar TCR, other pMHC", R.pp_similar_tcr_other_pmhc),
             ("same MHC, other peptide", R.pp_same_mhc_other_pep)]:
    print(f"  {k:28s} {int(c.sum())}")
print()
print("worst offenders (all chains >=70% to one pre-cutoff complex):")
print(R[R.rule3_strong].sort_values("rule3_min_id", ascending=False)[
    ["target", "rule3_ref", "rule3_min_id", "max_tcr_concat", "pep_id", "tm_full", "recommendation"]
    ].round(1).to_string(index=False))
