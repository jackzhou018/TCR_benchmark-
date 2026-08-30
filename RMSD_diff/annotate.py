#!/usr/bin/env python
"""Join af3_vs_esmfold2_rmsd.csv to the benchmark metadata and add the rigid-body
decomposition of the TCR disagreement -> RMSD_diff/divergence_with_metadata.csv.

    python RMSD_diff/annotate.py

tcr_rot_deg / tcr_com_shift: put both predictions in the MHC frame, then the rotation and
centre-of-mass offset that carries AF3's TCR onto ESMFold2's -- i.e. how differently the two
models dock the same TCR. tcr_internal_rmsd is the TCR fold on its own, for contrast.

<model>_rot_vs_native / <model>_com_vs_native: the same two numbers against the native instead
of against the other model, so a disagreement can be blamed on one model. <model>_tcr_height is
the TCR centre projected on the MHC-centre -> peptide-centre axis: negative means the TCR was
docked on the underside of the peptide-binding platform, the face the truncation exposes.
"""
import importlib.util, os
import gemmi
import numpy as np, pandas as pd

D = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(D)
spec = importlib.util.spec_from_file_location("m", f"{D}/af3_vs_esmfold2_rmsd.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def fit(a, b, keys):
    P, Q = np.array([a[k] for k in keys]), np.array([b[k] for k in keys])
    V, S, Wt = np.linalg.svd((P - P.mean(0)).T @ (Q - Q.mean(0)))
    return P.mean(0), Q.mean(0), V @ np.diag([1, 1, np.sign(np.linalg.det(V @ Wt))]) @ Wt


d = pd.read_csv(f"{D}/af3_vs_esmfold2_rmsd.csv").set_index("pdb_id")
nat = pd.read_csv(f"{ROOT}/input_data/natives/natives_manifest.csv").groupby("pdb_id").first()
ids = pd.read_csv(f"{ROOT}/input_data/PDB_IDs.csv").set_index("PDB ID")
sim = pd.read_csv(f"{ROOT}/analysis/training_sequence_similarity/"
                  "sequence_similarity_per_target.csv").set_index("pdb_id")
af = pd.read_csv(f"{ROOT}/models/AF3/results/af3-server_results.csv").set_index("pdb_id")
es = pd.read_csv(f"{ROOT}/models/ESMFold2/results/"
                 "esmfold2-fast-2026-05_loops10_steps100_results.csv").set_index("pdb_id")
gl = pd.read_csv(f"{ROOT}/output/DockQ/dockq_all.csv").groupby(
    ["model", "pdb_id"]).global_dockq.first().unstack(0)

d = d.join(nat[["method", "native_qc", "contacts_mhc_pep", "contacts_pep_tcr",
                "contacts_mhc_tcr", "contacts_tcr_tcr"]])
d = d.join(ids[["MHC allele", "Species", "Peptide sequence"]])
d = d.join(sim[["max_tcr_identity", "mhc_identity", "peptide_identity",
                "complex_mean_identity"]])
d["pep_len"] = d["Peptide sequence"].str.len()
d["af3_ptm"], d["af3_iptm"] = af.ptm, af.iptm
d["esm_plddt"], d["esm_iptm"] = es.mean_plddt, es.iptm
d["af3_dockq"], d["esm_dockq"] = gl.get("AF3"), gl.get("ESMFold2")

def nat_ca(pdb_id):
    """{(chain, i): xyz} for the native, keyed by position in the chain -- the native is
    cropped to the same FASTA as the predictions, so position i is the same residue, but
    its author numbering is not 1..N."""
    st = gemmi.read_structure(f"{ROOT}/input_data/natives/{pdb_id}.pdb")
    st.setup_entities(); st.remove_alternative_conformations()
    return {(ch.name, i): (a.pos.x, a.pos.y, a.pos.z) for ch in st[0]
            for i, a in enumerate([r.sole_atom("CA") for r in ch if r.find_atom("CA", "*")])}


def vs_native(pred, nat, mhc, tcr):
    """Superpose the prediction's MHC on the native's, then how far its TCR is from the
    native's: (rotation deg, centre-of-mass offset A, TCR height above the platform)."""
    k = sorted(set(pred) & set(nat))
    mk = [x for x in k if x[0] in mhc]
    tk = [x for x in k if x[0] in tcr]
    pc, qc, R = fit(pred, nat, mk)
    A = (np.array([pred[x] for x in tk]) - pc) @ R
    B = np.array([nat[x] for x in tk]) - qc
    _, _, R2 = fit(dict(zip(tk, A)), dict(zip(tk, B)), tk)
    # height is built from `pred` alone, so it needs no superposition
    M = np.array([pred[x] for x in mk]).mean(0)
    up = np.array([pred[x] for x in pred if x[0] == "B"]).mean(0) - M
    return (np.degrees(np.arccos(np.clip((np.trace(R2) - 1) / 2, -1, 1))),
            np.linalg.norm(A.mean(0) - B.mean(0)),
            float((np.array([pred[x] for x in tk]).mean(0) - M) @ (up / np.linalg.norm(up))))


rot, shift, internal = {}, {}, {}
nat_cmp = {}
for p, r in d.iterrows():
    a = m.ca(f"{m.AF3}/{p}/{p}_model_0.cif"); b = m.ca(f"{m.ESM}/{p}/{p}_model_0.cif")
    k = sorted(set(a) & set(b))
    mk = [x for x in k if x[0] in m.MHC[r.mhc_class]]
    tk = [x for x in k if x[0] in "".join(m.TCR[r.mhc_class].values())]
    pc, qc, R = fit(a, b, mk)                                    # MHC frame
    A = (np.array([a[x] for x in tk]) - pc) @ R
    B = np.array([b[x] for x in tk]) - qc
    _, _, R2 = fit(dict(zip(tk, A)), dict(zip(tk, B)), tk)       # TCR onto TCR within it
    rot[p] = np.degrees(np.arccos(np.clip((np.trace(R2) - 1) / 2, -1, 1)))
    shift[p] = np.linalg.norm(A.mean(0) - B.mean(0))
    internal[p] = m.rmsd([a[x] for x in tk], [b[x] for x in tk])
    n = nat_ca(p)
    ncls = {(c, i): v for (c, i), v in n.items()}
    # predictions number 1..N per chain; the native is keyed 0..N-1 in chain order
    off = lambda s: {(c, i - 1): v for (c, i), v in s.items()}
    nat_cmp[p] = (vs_native(off(a), ncls, m.MHC[r.mhc_class],
                            "".join(m.TCR[r.mhc_class].values())) +
                  vs_native(off(b), ncls, m.MHC[r.mhc_class],
                            "".join(m.TCR[r.mhc_class].values())))
d["tcr_rot_deg"], d["tcr_com_shift"], d["tcr_internal_rmsd"] = \
    pd.Series(rot), pd.Series(shift), pd.Series(internal)
d[["af3_rot_vs_native", "af3_com_vs_native", "af3_tcr_height",
   "esm_rot_vs_native", "esm_com_vs_native", "esm_tcr_height"]] = \
    pd.DataFrame(nat_cmp, index=range(6)).T

d.sort_values("mean_rmsd", ascending=False).round(3).to_csv(f"{D}/divergence_with_metadata.csv")
print(f"{len(d)} structures -> {D}/divergence_with_metadata.csv")
