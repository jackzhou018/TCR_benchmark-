"""Does the published RF run on ESMFold2 predictions?

Three questions: are its 14 inputs available; are the available ones in the range the
RF was trained on; and do the unavailable ones matter enough that imputing them is fatal.
"""
import importlib.util, joblib
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path("/14TBDrive/6TBDrive1_backup/benchmark_fresh")
spec = importlib.util.spec_from_file_location("CB", ROOT / "classifier_benchmark/run_benchmark.py")
CB = importlib.util.module_from_spec(spec); spec.loader.exec_module(CB)
rf = joblib.load(CB.MODEL_PATH)
af3 = pd.read_csv("feat_af3.csv"); esm = pd.read_csv("feat_esm.csv")

FROM_COORDS = {"global_plddt", "plddt_cdr1a", "plddt_cdr2a", "plddt_cdr3a",
               "plddt_cdr1b", "plddt_cdr2b", "plddt_cdr3b", "pdockq_AB"}
print("RF input availability for an ESMFold2 prediction as stored")
for f in CB.FEATURES:
    have = f in FROM_COORDS
    why = "pLDDT in B-factors / coordinates" if have else {
        "iptm_mean": "needs per-chain ipTM vector (only one global ipTM saved)",
        "iptm_tcrpmhc": "needs the chain-pair ipTM matrix",
        "avgipde": "needs the contact-probability matrix",
        "avgipae": "needs the PAE matrix",
        "avgpdockq2": "needs the PAE matrix",
        "iPSAE": "needs the PAE matrix"}[f]
    print(f"  {'YES' if have else 'NO ':4s} {f:14s} {why}")

print("\nimportance the RF puts on what is missing")
imp = pd.Series(rf.feature_importances_, index=CB.FEATURES).sort_values(ascending=False)
miss = [f for f in CB.FEATURES if f not in FROM_COORDS]
print("  missing features carry %.0f%% of total RF importance" % (100 * imp[miss].sum()))
print("  top 5:", ", ".join(f"{k} {v:.2f}" for k, v in imp.head(5).items()))

print("\nare the available inputs even in AF3's range? (mean +- sd)")
for f in sorted(FROM_COORDS):
    a, e = af3[f], esm[f]
    print(f"  {f:14s} AF3 {a.mean():6.2f}+-{a.std():5.2f}   ESMFold2 {e.mean():6.2f}+-{e.std():5.2f}"
          f"   shift {(e.mean()-a.mean())/a.std():+.2f} sd")

print("\nif the 6 missing inputs are drawn from AF3's own joint distribution,")
print("how much does the RF's answer move for a fixed ESMFold2 structure?")
rng = np.random.default_rng(0)
spread = []
for _, row in esm.iterrows():
    X = pd.DataFrame([{**{f: row[f] for f in FROM_COORDS},
                       **{f: af3[f].values[i] for f in miss}}
                      for i in rng.integers(0, len(af3), 200)])[CB.FEATURES]
    tiers = rf.predict_proba(X) @ np.array([CB.TIER_VALUE[c] for c in rf.classes_])
    spread.append(tiers.max() - tiers.min())
print(f"  expected-tier range across imputations: median {np.median(spread):.2f}, "
      f"90th pct {np.percentile(spread, 90):.2f}  (the tier scale is 0-1)")
