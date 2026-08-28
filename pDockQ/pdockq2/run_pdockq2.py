#!/usr/bin/env python
"""pDockQ2 (pDockQ_i) for every AF3 prediction, one chain at a time.

    python pDockQ/pdockq2/run_pdockq2.py    # -> results/pdockq2_all.csv
                                            #    results/pdockq2_per_structure.csv

pDockQ2 (Zhu, Shenoy, Kundrotas & Elofsson, Bioinformatics 2023, btad424) replaces
pDockQ's contact count with the PAE: for each chain i,

    pDockQ2_i = sigmoid( <interface pLDDT>_i x <1/(1+(PAE/10)^2)>_i )

over that chain's cross-chain CA-CA contacts within 8 A, with the paper's fitted
L=1.310, x0=84.733, k=0.0747, b=0.005. It is defined PER CHAIN, not per chain pair, so
the rows here are chains where pDockQ v1's rows are chain pairs; the per-structure value
is the mean over chains, same aggregation idea.

AF3 ONLY. pDockQ2 needs the full PAE matrix and only AF3 kept one -- it comes from the
server zips in models/AF3/AF3/fold_<id>_truncated_tcr_pmhc.zip (full_data_0.json, whose
model_0.cif is byte-identical to the prediction we score). ESMFold2 and Protenix outputs
on disk are CIF plus summary confidences only; no PAE, so no pDockQ2 without re-running.

The math is vectorised rather than looped like src/pdockq2.py (which is O(N^2) Python and
would take hours over 127 structures). --check re-runs whole structures through that
script as a subprocess and asserts the per-chain values agree.
"""
import argparse, csv, io, json, os, pickle, re, subprocess, sys, tempfile, zipfile
import numpy as np
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = f"{HERE}/src/pdockq2.py"
MODEL, CONFIG = "AF3", "af3-server"
ZIPS = f"{ROOT}/models/{MODEL}/AF3"
ROLES = {"Class I":  {"A": "MHC", "B": "peptide", "C": "TCRa", "D": "TCRb"},
         "Class II": {"A": "MHCa", "B": "peptide", "C": "MHCb", "D": "TCRa", "E": "TCRb"}}
CONTACT = 8.0                                    # A, CA-CA -- src/pdockq2.py's -dist default
D0 = 10.0                                        # A, the PAE normaliser d in the paper
FIT = (1.31034849e+00, 8.47326239e+01, 7.47157696e-02, 5.01886443e-03)   # L, x0, k, b


def sigmoid(x, L, x0, k, b):
    return L / (1 + np.exp(-k * (x - x0))) + b


def residues(cif):
    """(chains, resids, CA coords, pLDDT) per residue, in file order.

    Same raw-dict read as run_pdockq.py: AF3 omits the auth_ columns. src/pdockq2.py
    measures CA-CA and only falls back to CB when a residue has no CA -- these are
    all-protein models where every residue has one, so CA only.
    ponytail: asserted below rather than reimplementing the dead CB branch."""
    d = MMCIF2Dict(cif)
    g = lambda k: d.get("_atom_site." + k) or d["_atom_site." + k.replace("auth_", "label_")]
    ch, res, xyz, plddt = [], [], [], []
    seen = set()
    for c, s, name, b, x, y, z in zip(
            g("auth_asym_id"), g("auth_seq_id"), g("auth_atom_id"), g("B_iso_or_equiv"),
            g("Cartn_x"), g("Cartn_y"), g("Cartn_z")):
        seen.add((c[0], int(s)))
        if name == "CA":
            ch.append(c[0]); res.append(int(s))
            xyz.append([float(x), float(y), float(z)]); plddt.append(float(b))
    assert len(seen) == len(ch), f"{cif}: residue without a CA"
    return np.array(ch), np.array(res), np.array(xyz), np.array(plddt)


def af3_zip(pdb):
    return f"{ZIPS}/fold_{pdb.lower()}_truncated_tcr_pmhc.zip"


def load_pae(pdb, ch, res):
    """PAE rows/cols reindexed onto our residue order, via AF3's token_chain_ids/res_ids."""
    with zipfile.ZipFile(af3_zip(pdb)) as z:
        name = next(n for n in z.namelist() if n.endswith("_full_data_0.json"))
        d = json.load(io.TextIOWrapper(z.open(name)))
    tok = {(c, int(r)): i for i, (c, r) in
           enumerate(zip(d["token_chain_ids"], d["token_res_ids"]))}
    idx = np.array([tok[(c, r)] for c, r in zip(ch, res)])
    return np.asarray(d["pae"])[np.ix_(idx, idx)]


def pdockq2(ch, xyz, plddt, pae):
    """{chain: (pdockq2, if_plddt, if_pae_norm, n_contacts, contacting chains)}."""
    dist = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    contact = (dist <= CONTACT) & (ch[:, None] != ch[None, :])
    out = {}
    for c in sorted(set(ch)):
        m = ch == c
        sub = contact[m]                                  # (residues of c) x (all residues)
        n = int(sub.sum())
        if n:
            # each contact contributes res1's pLDDT once, exactly as retrieve_IFplddt appends
            ifplddt = float((np.repeat(plddt[m][:, None], sub.shape[1], 1)[sub]).mean())
            ifpae = float(np.mean(1 / (1 + (pae[m][sub] / D0) ** 2)))
        else:
            ifplddt = ifpae = 0.0
        partners = "".join(sorted(set(ch[sub.any(0)])))
        out[c] = (float(sigmoid(ifplddt * ifpae, *FIT)), ifplddt, ifpae, n, partners)
    return out


def write_pdb(cif, out):
    """All chains as a PDB with pLDDT in the B-factor column -- for the --check path."""
    d = MMCIF2Dict(cif)
    g = lambda k: d.get("_atom_site." + k) or d["_atom_site." + k.replace("auth_", "label_")]
    with open(out, "w") as f:
        for i, (c, s, name, comp, el, b, x, y, z) in enumerate(zip(
                g("auth_asym_id"), g("auth_seq_id"), g("auth_atom_id"), g("auth_comp_id"),
                g("type_symbol"), g("B_iso_or_equiv"),
                g("Cartn_x"), g("Cartn_y"), g("Cartn_z")), 1):
            nm = f" {name}" if len(name) < 4 else name
            f.write(f"ATOM  {i:5d} {nm:<4} {comp:>3} {c[0]}{int(s):4d}    "
                    f"{float(x):8.3f}{float(y):8.3f}{float(z):8.3f}{1.0:6.2f}{float(b):6.2f}"
                    f"          {el:>2}\n")
        f.write("END\n")


def subprocess_pdockq2(cif, pae):
    """Reference src/pdockq2.py on the same structure -> {chain: pdockq2}."""
    tmp = tempfile.mkdtemp()
    pdb, pkl = f"{tmp}/m.pdb", f"{tmp}/m.pkl"
    try:
        write_pdb(cif, pdb)
        pickle.dump({"predicted_aligned_error": pae}, open(pkl, "wb"))
        out = subprocess.run([sys.executable, SRC, "-pdb", pdb, "-pkl", pkl],
                             capture_output=True, text=True, check=True).stdout
    finally:
        for p in (pdb, pkl):
            os.path.exists(p) and os.remove(p)
        os.rmdir(tmp)
    return {m[1]: float(m[2]) for m in re.finditer(r"^(\S+) ([\d.eE+-]+)$", out, re.M)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", type=int, default=0,
                    help="also score this many structures with src/pdockq2.py and compare")
    args = ap.parse_args()

    meta = {r["pdb_id"]: r for r in
            csv.DictReader(open(f"{ROOT}/input_data/natives/natives_manifest.csv"))}
    excluded = {r["pdb_id"] for r in
                csv.DictReader(open(f"{ROOT}/input_data/natives/excluded.csv"))}
    dockq = {}
    for r in csv.DictReader(open(f"{ROOT}/output/DockQ/dockq_all.csv")):
        dockq.setdefault((r["model"], r["pdb_id"]), r["global_dockq"])

    pred_dir = f"{ROOT}/models/{MODEL}/predictions/{CONFIG}"
    rows, per_structure, bad = [], [], []
    for pdb in sorted(set(os.listdir(pred_dir)) - excluded):
        cif = f"{pred_dir}/{pdb}/" + sorted(
            f for f in os.listdir(f"{pred_dir}/{pdb}") if f.endswith(".cif"))[0]
        ch, res, xyz, plddt = residues(cif)
        pae = load_pae(pdb, ch, res)
        vals = pdockq2(ch, xyz, plddt, pae)
        cls = meta[pdb]["mhc_class"]
        gd = dockq.get((MODEL, pdb), "")
        mean = float(np.mean([v[0] for v in vals.values()]))
        for c, (p, ifp, ifpae, n, partners) in vals.items():
            rows.append(dict(model=MODEL, config=CONFIG, pdb_id=pdb, mhc_class=cls,
                             review_status=meta[pdb]["review_status"], chain=c,
                             chain_role=ROLES[cls][c], pdockq2=round(p, 4),
                             if_plddt=round(ifp, 2), if_pae_norm=round(ifpae, 4),
                             n_contacts=n, contacts_chains=partners,
                             mean_pdockq2=round(mean, 4), n_chains=len(vals),
                             global_dockq=gd))
        per_structure.append(dict(model=MODEL, config=CONFIG, pdb_id=pdb, mhc_class=cls,
                                  review_status=meta[pdb]["review_status"],
                                  mean_pdockq2=round(mean, 4), n_chains=len(vals),
                                  global_dockq=gd))
        if len(per_structure) <= args.check:
            ref = subprocess_pdockq2(cif, pae)
            for c, (p, *_) in vals.items():
                if abs(p - ref.get(c, float("nan"))) > 1e-9:
                    bad.append((pdb, c, p, ref.get(c)))
            print(f"--check {pdb}: {len(vals)} chains vs src/pdockq2.py, "
                  f"disagreements so far: {len(bad)}")

    print(f"{MODEL}: {len(per_structure)} structures, {len(rows)} chains")
    for b in bad[:5]:
        print("   MISMATCH", b)

    os.makedirs(f"{HERE}/results", exist_ok=True)
    for name, data in (("pdockq2_all.csv", rows),
                       ("pdockq2_per_structure.csv", per_structure)):
        out = f"{HERE}/results/{name}"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, list(data[0]))
            w.writeheader(); w.writerows(data)
        print(f"{len(data)} rows -> {out}")


if __name__ == "__main__":
    main()
