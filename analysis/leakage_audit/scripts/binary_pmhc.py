"""Partial prior 4d: was the peptide already in a pre-cutoff pMHC structure with no TCR?

RCSB search: entries released <= 2021-09-30 whose MHC entity matches the target's MHC at
>=95% and that have <= 3 polymer entities (MHC + b2m + peptide, i.e. no TCR pair).
Then the peptide entity sequences of those entries are compared to the target peptide.
"""
import json, time, urllib.request, urllib.error
from pathlib import Path
import pandas as pd
from Bio import Align
from Bio.Align import substitution_matrices

HERE = Path(__file__).parent; ROOT = Path("/14TBDrive/6TBDrive1_backup/benchmark_fresh")
SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
GRAPHQL = "https://data.rcsb.org/graphql"
al = Align.PairwiseAligner(mode="global", substitution_matrix=substitution_matrices.load("BLOSUM62"),
                           open_gap_score=-11, extend_gap_score=-1)
def ident(a, b):
    if not a or not b: return 0.0
    aln = al.align(a, b)[0]; ta, qa = aln.aligned
    m = sum(sum(1 for x, y in zip(a[s1:e1], b[s2:e2]) if x == y) for (s1, e1), (s2, e2) in zip(ta, qa))
    return 100.0 * m / min(len(a), len(b))

def post(url, payload, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, json.dumps(payload).encode(),
                                         {"Content-Type": "application/json"})
            return json.load(urllib.request.urlopen(req, timeout=90))
        except urllib.error.HTTPError as e:
            if e.code == 204: return {"result_set": []}
            if i == tries - 1: raise
        except Exception:
            if i == tries - 1: raise
        time.sleep(2)

per = pd.read_csv(ROOT / "analysis/training_sequence_similarity/sequence_similarity_per_target.csv")
ROLE_I = {"A": "mhc", "B": "pep"}; ROLE_II = {"A": "mhc", "B": "pep", "C": "mhc2"}
seq = {}
for r in per.itertuples():
    f, k = {}, None
    for line in open(ROOT / f"input_data/truncated_structures/fastas/{r.pdb_id}_truncated.fasta"):
        line = line.strip()
        if line.startswith(">"): k = line[1:].replace("Chain_", ""); f[k] = ""
        elif k: f[k] += line
    roles = ROLE_I if r.mhc_class == "Class I" else ROLE_II
    seq[r.pdb_id] = {roles[c]: s for c, s in f.items() if c in roles}

def query(mhc_seq):
    return {"query": {"type": "group", "logical_operator": "and", "nodes": [
        {"type": "terminal", "service": "sequence", "parameters": {
            "evalue_cutoff": 1e-6, "identity_cutoff": 0.95, "sequence_type": "protein",
            "value": mhc_seq}},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_accession_info.initial_release_date", "operator": "less_or_equal",
            "value": "2021-09-30"}},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_entry_info.polymer_entity_count", "operator": "less_or_equal",
            "value": 3}}]},
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": 1000}, "results_verbosity": "compact"}}

uniq = {}
for t, s in seq.items():
    uniq.setdefault(s["mhc"], []).append(t)
print(f"{len(uniq)} distinct MHC chain-A sequences across {len(seq)} targets", flush=True)
hits = {}
for i, (mseq, targets) in enumerate(uniq.items(), 1):
    try:
        res = post(SEARCH, query(mseq))
        ids = res.get("result_set", []) or []
    except Exception as e:
        print("  search failed for", targets[0], e, flush=True); ids = []
    for t in targets: hits[t] = ids
    print(f"  [{i}/{len(uniq)}] {targets[0]}: {len(ids)} pre-cutoff non-ternary entries", flush=True)
json.dump(hits, open(HERE / "binary_hits.json", "w"))

allids = sorted({i for v in hits.values() for i in v})
print(f"{len(allids)} unique candidate entries; fetching peptide sequences", flush=True)
ent = {}
for k in range(0, len(allids), 50):
    batch = allids[k:k + 50]
    q = ('{entries(entry_ids:%s){rcsb_id polymer_entities{'
         'entity_poly{pdbx_seq_one_letter_code_can}}}}' % json.dumps(batch))
    d = post(GRAPHQL, {"query": q})
    for e in d["data"]["entries"]:
        ent[e["rcsb_id"]] = [(pe["entity_poly"]["pdbx_seq_one_letter_code_can"] or "").replace("\n", "")
                             for pe in e["polymer_entities"]]
print(f"fetched {len(ent)} entries", flush=True)

rows = []
for t, ids in hits.items():
    pep = seq[t]["pep"]
    best, bestid = 0.0, None
    for i in ids:
        for s in ent.get(i, []):
            if 5 <= len(s) <= 25:
                v = ident(pep, s)
                if v > best: best, bestid = v, i
    rows.append({"target": t, "n_binary_candidates": len(ids),
                 "binary_pep_identity": best, "binary_pep_ref": bestid,
                 "binary_pmhc_prior": best >= 90})
B = pd.DataFrame(rows)
B.to_csv(HERE / "binary_pmhc.csv", index=False)
print("targets with an exact/near-exact pre-cutoff pMHC (no TCR):", int(B.binary_pmhc_prior.sum()))
