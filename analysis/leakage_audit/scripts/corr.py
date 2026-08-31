"""Does closeness to the pre-cutoff record predict how well each model does?"""
import sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, "/14TBDrive/6TBDrive1_backup/benchmark_fresh/paper/figures/scripts")
from style import keep, load_dockq, global_dockq

A = pd.read_csv("audit_final.csv")
d = global_dockq(keep(load_dockq()))
d = d.merge(A, left_on="pdb_id", right_on="target")
MEAS = [("max_tcr_concat", "concatenated TCR Vα+Vβ identity"),
        ("max_tcra", "TCRα identity"), ("max_tcrb", "TCRβ identity"),
        ("pep_id", "peptide identity (closest complex)"),
        ("best_mhc_id", "MHC identity"),
        ("closest_mean_id", "mean identity, closest complex"),
        ("rule3_min_id", "worst chain vs best all-chain match"),
        ("tm_full", "multimer TM-score")]
MODELS = ["AF3", "Protenix", "ESMFold2", "TCRmodel2"]
rng = np.random.default_rng(0)

def boot(x, y, n=10000):
    v = [stats.spearmanr(x[i], y[i]).statistic for i in (rng.integers(0, len(x), len(x)) for _ in range(n))]
    return np.percentile(v, [2.5, 97.5])

rows = []
for m in MODELS:
    g = d[d.model == m]
    for col, lab in MEAS:
        s = g.dropna(subset=[col])
        if len(s) < 10: continue
        rho = stats.spearmanr(s[col], s.global_dockq)
        lo, hi = boot(s[col].values, s.global_dockq.values)
        rows.append({"model": m, "measure": lab, "col": col, "n": len(s),
                     "rho": rho.statistic, "p": rho.pvalue, "lo": lo, "hi": hi})
C = pd.DataFrame(rows)
C.to_csv("identity_vs_dockq.csv", index=False)
for m in MODELS:
    print(f"\n{m}")
    for r in C[C.model == m].itertuples():
        star = "*" if (r.lo > 0 or r.hi < 0) else " "
        print(f"  {star} {r.measure:38s} n={r.n:3d}  rho={r.rho:+.3f} [{r.lo:+.3f},{r.hi:+.3f}]  p={r.p:.3g}")

print("\n--- accuracy by audit class (median Global DockQ) ---")
d["klass"] = np.where(d.rule3_strong, "near-duplicate (n=10)",
             np.where(d.rule2_violation, "TCR-redundant (n=32)",
             np.where(d.rule3_violation | d.tm_violation, "marginal (n=42)", "clean (n=42)")))
piv = d.pivot_table(index="klass", columns="model", values="global_dockq", aggfunc="median")
print(piv.round(3).to_string())
print("\n--- near-duplicate vs clean, paired by nothing (independent groups) ---")
for m in MODELS:
    g = d[d.model == m]
    a = g[g.klass.str.startswith("near")].global_dockq
    b = g[g.klass.str.startswith("clean")].global_dockq
    if len(a) < 3 or len(b) < 3: continue
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    diff = a.median() - b.median()
    bs = [np.median(rng.choice(a, len(a))) - np.median(rng.choice(b, len(b))) for _ in range(10000)]
    print(f"  {m:10s} near-dup {a.median():.3f} (n={len(a)}) vs clean {b.median():.3f} (n={len(b)})"
          f"   Δ={diff:+.3f} [{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}]  p={u.pvalue:.3f}")
d.to_csv("dockq_with_audit.csv", index=False)
