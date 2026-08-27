#!/usr/bin/env python
"""Pick the 30-structure sample, once, reproducibly.

Pool = the benchmark's 126 scoreable complexes (READY, native built, not in
excluded.csv) that also have an AlphaFold Server result zip -- the zip is
required because every track reuses the server's MSAs (models/AF3/AF3/*.zip).
Row order follows manifest.csv, per ../CLAUDE.md.
"""
import csv, glob, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEED = 20260827          # frozen; changing it changes the experiment
N = 30

rows = list(csv.DictReader(open(f"{ROOT}/input_data/truncated_structures/manifest.csv")))
order = [r["PDB ID"] for r in rows]
ready = {r["PDB ID"]: r for r in rows if r["Status"] == "READY"}
zips = {os.path.basename(p).split("_")[1].upper() for p in glob.glob(f"{ROOT}/models/AF3/AF3/*.zip")}
excl = {r["pdb_id"] for r in csv.DictReader(open(f"{ROOT}/input_data/natives/excluded.csv"))}
natives = {os.path.basename(p)[:-4] for p in glob.glob(f"{ROOT}/input_data/natives/*.pdb")}

pool = [p for p in order if p in ready and p in zips and p in natives and p not in excl]
assert len(pool) == 126, len(pool)

import random
sample = sorted(random.Random(SEED).sample(pool, N), key=order.index)

def n_res(pdb):
    n, seq = 0, open(f"{ROOT}/input_data/truncated_structures/fastas/{pdb}_truncated.fasta")
    return sum(len(l.strip()) for l in seq if not l.startswith(">"))

out = f"{ROOT}/overnight/sample.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["pdb_id", "mhc_class", "n_residues", "resolution", "review_status"])
    for p in sample:
        r = ready[p]
        w.writerow([p, r["MHC class"], n_res(p), r["Resolution"], r["Status"]])
print(f"seed={SEED} pool={len(pool)} -> {N} written to {out}")
print(" ".join(sample))
