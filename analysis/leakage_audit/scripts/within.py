"""Test 6: redundancy inside the benchmark itself."""
from pathlib import Path
import itertools
import numpy as np, pandas as pd
from Bio import Align
from Bio.Align import substitution_matrices
ROOT = Path("/14TBDrive/6TBDrive1_backup/benchmark_fresh"); HERE = Path(__file__).parent
al = Align.PairwiseAligner(mode="global", substitution_matrix=substitution_matrices.load("BLOSUM62"),
                           open_gap_score=-11, extend_gap_score=-1,
                           target_end_gap_score=0.0, query_end_gap_score=0.0)
def ident(a, b):
    if not a or not b: return 0.0, 0.0
    aln = al.align(a, b)[0]; ta, qa = aln.aligned
    m = sum(sum(1 for x, y in zip(a[s1:e1], b[s2:e2]) if x == y) for (s1, e1), (s2, e2) in zip(ta, qa))
    return 100.0 * m / min(len(a), len(b)), min(sum(e-s for s,e in ta)/len(a), sum(e-s for s,e in qa)/len(b))
def mismatches(a, b):
    if len(a) != len(b): return None
    return sum(1 for x, y in zip(a, b) if x != y)

per = pd.read_csv(ROOT / "analysis/training_sequence_similarity/sequence_similarity_per_target.csv")
ROLE_I = {"A": "mhc", "B": "pep", "C": "tcra", "D": "tcrb"}
ROLE_II = {"A": "mhc", "B": "pep", "C": "mhc2", "D": "tcra", "E": "tcrb"}
seq = {}
for r in per.itertuples():
    f, k = {}, None
    for line in open(ROOT / f"input_data/truncated_structures/fastas/{r.pdb_id}_truncated.fasta"):
        line = line.strip()
        if line.startswith(">"): k = line[1:].replace("Chain_", ""); f[k] = ""
        elif k: f[k] += line
    roles = ROLE_I if r.mhc_class == "Class I" else ROLE_II
    seq[r.pdb_id] = {roles[c]: s for c, s in f.items() if c in roles}
cls = per.set_index("pdb_id").mhc_class
allele = per.set_index("pdb_id").mhc_allele

pairs = []
ids = list(per.pdb_id)
for a, b in itertools.combinations(ids, 2):
    if cls[a] != cls[b]:
        continue
    A, B = seq[a], seq[b]
    ca, cova = ident(A["tcra"] + A["tcrb"], B["tcra"] + B["tcrb"])
    ia, _ = ident(A["tcra"], B["tcra"]); ib, _ = ident(A["tcrb"], B["tcrb"])
    mh, _ = ident(A["mhc"], B["mhc"])
    if "mhc2" in A and "mhc2" in B:
        mh2, _ = ident(A["mhc2"], B["mhc2"]); mh = (mh + mh2) / 2
    pe, _ = ident(A["pep"], B["pep"])
    same_pmhc = mh >= 95
    rule = (ca >= 90 and cova >= .80) or (max(ia, ib) >= 95 and same_pmhc)
    if rule:
        pairs.append({"a": a, "b": b, "tcr_concat": ca, "tcra": ia, "tcrb": ib,
                      "mhc": mh, "peptide": pe, "same_allele": allele[a] == allele[b],
                      "pep_mismatches": mismatches(A["pep"], B["pep"]),
                      "paired_variant": bool(max(ia, ib) >= 95 and same_pmhc and
                                             (mismatches(A["pep"], B["pep"]) or 99) <= 2)})
P = pd.DataFrame(pairs)
P.to_csv(HERE / "within_pairs.csv", index=False)

parent = {i: i for i in ids}
def find(x):
    while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
    return x
for r in P.itertuples():
    ra, rb = find(r.a), find(r.b)
    if ra != rb: parent[ra] = rb
comp = {}
for i in ids: comp.setdefault(find(i), []).append(i)
clusters = sorted(comp.values(), key=lambda v: (-len(v), v[0]))
lab = {t: f"C{n+1:02d}" for n, c in enumerate(clusters) for t in c}
pd.DataFrame([{"target": t, "cluster": lab[t],
               "cluster_size": len(comp[find(t)])} for t in ids]).to_csv(HERE / "clusters.csv", index=False)
print(f"redundant pairs           {len(P)}  (of which paired peptide variants: {int(P.paired_variant.sum())})")
print(f"clusters                  {len(clusters)}   (independent cases after dedup)")
print(f"targets in a cluster >1   {sum(len(c) for c in clusters if len(c) > 1)}")
print("largest clusters:")
for c in clusters[:8]:
    if len(c) > 1: print("  ", len(c), " ".join(c))
