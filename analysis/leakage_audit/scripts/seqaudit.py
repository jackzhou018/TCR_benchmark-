"""Sequence side of the leakage audit: tests 1, 2, 3, 4 and 6.

Targets  : the 126 scoreable benchmark complexes, chains as cropped in
           input_data/truncated_structures/fastas (A/B[/C] pMHC, TCRa, TCRb).
Reference: the 259 pre-cutoff (<= 2021-09-30) TCR-pMHC complexes assembled from
           TCR3d + STCRDab in analysis/training_sequence_similarity/, cropped to the
           same domain boundaries.
Identity : global Needleman-Wunsch (BLOSUM62, gap -11/-1), identity = matches over
           the shorter sequence, with coverage = aligned span / length for both.
"""
import itertools, json, pickle
from pathlib import Path
import numpy as np, pandas as pd
from Bio import Align
from Bio.Align import substitution_matrices

ROOT = Path("/14TBDrive/6TBDrive1_backup/benchmark_fresh")
HERE = Path(__file__).parent
COV = 0.80

aligner = Align.PairwiseAligner(mode="global", substitution_matrix=substitution_matrices.load("BLOSUM62"),
                                open_gap_score=-11, extend_gap_score=-1,
                                target_end_gap_score=0.0, query_end_gap_score=0.0)

def identity(a, b):
    """(% identity over the shorter sequence, coverage of a, coverage of b)."""
    if not a or not b:
        return 0.0, 0.0, 0.0
    aln = aligner.align(a, b)[0]
    ta, qa = aln.aligned
    m = sum(sum(1 for x, y in zip(a[s1:e1], b[s2:e2]) if x == y)
            for (s1, e1), (s2, e2) in zip(ta, qa))
    span_a = sum(e - s for s, e in ta); span_b = sum(e - s for s, e in qa)
    return 100.0 * m / min(len(a), len(b)), span_a / len(a), span_b / len(b)

# ---------- targets
per = pd.read_csv(ROOT / "analysis/training_sequence_similarity/sequence_similarity_per_target.csv")
TARGETS = per[["pdb_id", "mhc_class", "mhc_allele", "species", "peptide", "release_date",
               "resolution", "review_status"]].copy()

def read_fasta(p):
    out, k = {}, None
    for line in open(p):
        line = line.strip()
        if line.startswith(">"):
            k = line[1:].replace("Chain_", ""); out[k] = ""
        elif k:
            out[k] += line
    return out

ROLE_I = {"A": "MHC_I_alpha1_alpha2", "B": "peptide", "C": "TCR_alpha_variable", "D": "TCR_beta_variable"}
ROLE_II = {"A": "MHC_II_alpha1", "B": "peptide", "C": "MHC_II_beta1",
           "D": "TCR_alpha_variable", "E": "TCR_beta_variable"}
tseq = {}
for r in TARGETS.itertuples():
    f = read_fasta(ROOT / f"input_data/truncated_structures/fastas/{r.pdb_id}_truncated.fasta")
    roles = ROLE_I if r.mhc_class == "Class I" else ROLE_II
    tseq[r.pdb_id] = {roles[c]: s for c, s in f.items() if c in roles}

# ---------- references
ref = pd.read_csv(ROOT / "analysis/training_sequence_similarity/training_reference_set.csv")
assert (ref.release_date <= "2021-09-30").all(), "reference set is not all pre-cutoff"
rseq, rmeta = {}, {}
for pid, g in ref.groupby("pdb_id"):
    rseq[pid] = {row.role: row.seq for row in g.itertuples()}
    rmeta[pid] = {"release_date": g.release_date.iloc[0], "mhc_class": g.mhc_class.iloc[0],
                  "complete": bool(g.complete_complex.iloc[0]),
                  "chains": {row.role: row.auth_chains for row in g.itertuples()}}

# ---------- all target x reference identities, per role
rows = []
for t in TARGETS.pdb_id:
    ts = tseq[t]
    for rp, rs in rseq.items():
        if rmeta[rp]["mhc_class"] != TARGETS.set_index("pdb_id").mhc_class[t]:
            continue                      # class I vs class II are not corresponding chains
        rec = {"target": t, "ref": rp}
        for role in ts:
            if role not in rs:
                rec[role] = np.nan; rec[role + "_cov"] = 0.0; continue
            pid_, ca, cb = identity(ts[role], rs[role])
            rec[role] = pid_
            rec[role + "_cov"] = min(ca, cb)
        rows.append(rec)
M = pd.DataFrame(rows)
M.to_csv(HERE / "pairwise_target_vs_ref.csv", index=False)
print("pairwise matrix:", M.shape)

# ---------- concatenated TCR identity, same reference complex
def concat_tcr(t, rp):
    a, b = tseq[t].get("TCR_alpha_variable", ""), tseq[t].get("TCR_beta_variable", "")
    ra, rb = rseq[rp].get("TCR_alpha_variable", ""), rseq[rp].get("TCR_beta_variable", "")
    if not (a and b and ra and rb):
        return np.nan, 0.0
    pid_, ca, cb = identity(a + b, ra + rb)
    return pid_, min(ca, cb)

cc = []
for t in TARGETS.pdb_id:
    cls = TARGETS.set_index("pdb_id").mhc_class[t]
    for rp in rseq:
        if rmeta[rp]["mhc_class"] != cls:
            continue
        v, cov = concat_tcr(t, rp)
        cc.append({"target": t, "ref": rp, "tcr_concat": v, "tcr_concat_cov": cov})
C = pd.DataFrame(cc)
C.to_csv(HERE / "tcr_concat_vs_ref.csv", index=False)
print("concat TCR matrix:", C.shape)
