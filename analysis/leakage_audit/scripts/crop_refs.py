"""Crop each pre-cutoff reference entry to the same domains as the benchmark natives,
relabelled to benchmark chain letters, so Foldseek-Multimer compares like with like."""
from pathlib import Path
import pandas as pd
from Bio.PDB import MMCIFParser, PDBIO, Select
from Bio.PDB.Polypeptide import is_aa

ROOT = Path("/14TBDrive/6TBDrive1_backup/benchmark_fresh")
HERE = Path(__file__).parent
OUT = HERE / "ref_pdb"; OUT.mkdir(exist_ok=True)
LETTER_I = {"MHC_I_alpha1_alpha2": "A", "peptide": "B",
            "TCR_alpha_variable": "C", "TCR_beta_variable": "D"}
LETTER_II = {"MHC_II_alpha1": "A", "peptide": "B", "MHC_II_beta1": "C",
             "TCR_alpha_variable": "D", "TCR_beta_variable": "E"}
ref = pd.read_csv(ROOT / "analysis/training_sequence_similarity/training_reference_set.csv")
parser = MMCIFParser(QUIET=True)
io = PDBIO()
ok = bad = 0
for pid, g in ref.groupby("pdb_id"):
    letters = LETTER_I if g.mhc_class.iloc[0] == "Class I" else LETTER_II
    try:
        model = parser.get_structure(pid, str(HERE / "ref_cif" / f"{pid}.cif"))[0]
    except Exception as e:
        print("parse fail", pid, e); bad += 1; continue
    keep = {}
    for row in g.itertuples():
        if row.role not in letters or not isinstance(row.auth_chains, str):
            continue
        cid = row.auth_chains.split(";")[0].split(",")[0].strip()
        if cid not in model:
            continue
        lo, hi = (int(x) for x in str(row.crop_span).split("-")[:2]) if "-" in str(row.crop_span) else (-10**6, 10**6)
        res = [r for r in model[cid] if is_aa(r, standard=True) and lo <= r.id[1] <= hi]
        if res:
            keep[letters[row.role]] = res
    if len(keep) != len(letters):
        bad += 1; continue
    with open(OUT / f"{pid}.pdb", "w") as fh:
        n = 0
        for letter in letters.values():
            for r in keep[letter]:
                for a in r:
                    if a.element == "H":
                        continue
                    n += 1
                    x, y, z = a.coord
                    fh.write(f"ATOM  {n:5d} {a.get_name():<4.4s}{' '}{r.resname:>3s} {letter}"
                             f"{r.id[1]:4d}{r.id[2].strip():1s}   {x:8.3f}{y:8.3f}{z:8.3f}"
                             f"{1.0:6.2f}{0.0:6.2f}          {a.element:>2s}\n")
            fh.write("TER\n")
        fh.write("END\n")
    ok += 1
print(f"cropped {ok} references, {bad} skipped (incomplete or unparsable)")
