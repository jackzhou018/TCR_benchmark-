"""Merge every test into one row per target, with recommendations."""
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).parent
ROOT = Path("/14TBDrive/6TBDrive1_backup/benchmark_fresh")
R = pd.read_csv(HERE / "rules.csv")
CL = pd.read_csv(HERE / "clusters.csv")
per = pd.read_csv(ROOT / "analysis/training_sequence_similarity/sequence_similarity_per_target.csv")

# ---- foldseek multimer report
fs = pd.read_csv(HERE / "fs_aln_report", sep="\t", header=None,
                 names=["query", "target", "qchains", "tchains", "qtm", "ttm", "u", "t", "assign"])
fs = fs[fs["query"].isin(R.target)]
need = per.set_index("pdb_id").mhc_class.map({"Class I": 4, "Class II": 5})
fs["nchain"] = fs.qchains.str.count(",") + 1
fs["identity_map"] = fs.qchains == fs.tchains
fs["complete"] = fs.apply(lambda r: r.nchain == need[r["query"]], axis=1)
best = fs.sort_values("qtm", ascending=False).groupby("query").first()
bestc = (fs[fs.complete & fs.identity_map].sort_values("qtm", ascending=False)
         .groupby("query").first())
R["tm_best"] = R.target.map(best.qtm)
R["tm_best_ref"] = R.target.map(best.target)
R["tm_best_nchain"] = R.target.map(best.nchain)
R["tm_best_map_ok"] = R.target.map(best.identity_map & best.complete)
R["tm_full"] = R.target.map(bestc.qtm)
R["tm_full_ref"] = R.target.map(bestc.target)
R["tm_violation"] = R.tm_full.fillna(0) > 0.90

# ---- severity split on rule 3 (the >40% threshold is loose; >=70% on every chain is not)
R["rule3_strong"] = R.rule3_violation & (R.rule3_min_id >= 70)
R["partial_prior"] = np.where(
    R.pp_same_pmhc, "same/similar pMHC",
    np.where(R.pp_similar_tcr_other_pmhc, "similar TCR, other pMHC",
             np.where(R.pp_same_mhc_other_pep, "same MHC, other peptide", "none")))
R = R.merge(CL, on="target")

def recommend(r):
    if r.temporal_violation:
        return "Exclude—temporal leakage"
    if r.rule3_strong or r.tm_violation:
        return "Exclude—complete-complex similarity"
    if r.rule2_violation:
        return "Exclude—high TCR redundancy"
    if r.rule3_violation:
        return "Retain in standard benchmark; exclude from strict subset"
    if r.cluster_size > 1:
        return "Combine with another target as one cluster"
    if r.partial_prior != "none":
        return "Retain—temporally blind with partial prior"
    return "Retain—strongly blind"
R["recommendation"] = R.apply(recommend, axis=1)

def rule_cited(r):
    bits = []
    if r.temporal_violation: bits.append(f"rule 1 (release {r.release_date_rcsb})")
    if r.rule2_violation: bits.append("rule 2 " + r.rule2_hits)
    if r.rule3_violation: bits.append(f"rule 3 all-chain>40% vs {r.rule3_ref} (min {r.rule3_min_id:.0f}%)")
    if r.tm_violation: bits.append(f"rule 5 TM {r.tm_full:.2f} vs {r.tm_full_ref}")
    return "; ".join(bits)
R["rules_triggered"] = R.apply(rule_cited, axis=1)
R.to_csv(HERE / "audit_full.csv", index=False)

n = len(R)
print(f"targets                                   {n}")
print(f"1  failing temporal cutoff                {int(R.temporal_violation.sum())}")
print(f"2  violating 90/95% TCR rule              {int(R.rule2_violation.sum())}")
print(f"3  violating all-chain >40% rule          {int(R.rule3_violation.sum())}"
      f"   (of these, >=70% on every chain: {int(R.rule3_strong.sum())})")
print(f"5  multimer TM > 0.90 (complete, correct) {int(R.tm_violation.sum())}"
      f"   [any chain subset: {int((R.tm_best > 0.90).sum())}]")
clean = R[~(R.rule2_violation | R.rule3_violation | R.tm_violation)]
print(f"4  clean of 2/3/5                         {len(clean)}"
      f"   with only partial priors: {int((clean.partial_prior != 'none').sum())}"
      f"   with none: {int((clean.partial_prior == 'none').sum())}")
print(f"6  independent clusters                   {R.cluster.nunique()}")
print()
print(R.recommendation.value_counts().to_string())
