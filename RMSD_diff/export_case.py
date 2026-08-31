#!/usr/bin/env python
"""Drop one complex's native + both predictions into RMSD_diff/<PDB>/ as mmCIF.

    python RMSD_diff/export_case.py 7SG2

The predictions are rigid-body moved onto the native's MHC chain(s) (CA, class-I A /
class-II A+C) so the three files open in the same frame and the peptide/TCR differences are
what you see. Nothing else about them is touched.
"""
import difflib, importlib.util, os, sys
import gemmi, numpy as np

D = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(D)
spec = importlib.util.spec_from_file_location("m", f"{D}/af3_vs_esmfold2_rmsd.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

pdb_id = sys.argv[1]
out = f"{D}/{pdb_id}"; os.makedirs(out, exist_ok=True)
native = gemmi.read_structure(f"{ROOT}/input_data/natives/{pdb_id}.pdb")
native.setup_entities(); native.remove_alternative_conformations()   # 8RYP etc. are high enough
                                                                     # resolution to have altlocs
cls = "Class II" if any(ch.name == "E" for ch in native[0]) else "Class I"
mhc = m.MHC[cls]


def cas(st):
    """{chain: ([CA position, ...], one-letter sequence)} in file order. The native is cropped
    to the same FASTA as the prediction, so position i is usually the same residue -- but a
    native with disordered residues is missing some, hence the sequence for `pair()`."""
    return {ch.name: ([r.sole_atom("CA").pos for r in ch if r.find_atom("CA", "*")],
                      "".join(gemmi.find_tabulated_residue(r.name).one_letter_code.upper()
                              for r in ch if r.find_atom("CA", "*")))
            for ch in st[0]}


def pair(a, b):
    """Index pairs of identical residues, in order -- exact when the two chains are equal
    length, and skipping the native's gaps when they are not (9RU5 chain A is short by 4)."""
    return [(i + k, j + k) for i, j, n in
            difflib.SequenceMatcher(None, a[1], b[1], autojunk=False).get_matching_blocks()
            for k in range(n)]


nat = cas(native)
for tag, path in (("AF3", f"{m.AF3}/{pdb_id}/{pdb_id}_model_0.cif"),
                  ("ESMFold2", f"{m.ESM}/{pdb_id}/{pdb_id}_model_0.cif")):
    st = gemmi.read_structure(path); st.setup_entities()
    st.remove_alternative_conformations()
    pred = cas(st)
    P, Q = [], []
    for c in mhc:
        ij = pair(pred[c], nat[c])
        assert len(ij) >= .9 * len(nat[c][1]), f"{pdb_id} chain {c}: only {len(ij)} matched"
        if len(ij) != len(nat[c][1]):
            print(f"  {c}: native {len(nat[c][1])} res, {len(ij)} matched to the prediction")
        P += [[pred[c][0][i].x, pred[c][0][i].y, pred[c][0][i].z] for i, _ in ij]
        Q += [[nat[c][0][j].x, nat[c][0][j].y, nat[c][0][j].z] for _, j in ij]
    P, Q = np.array(P), np.array(Q)
    pc, qc = P.mean(0), Q.mean(0)
    V, S, Wt = np.linalg.svd((P - pc).T @ (Q - qc))
    R = V @ np.diag([1, 1, np.sign(np.linalg.det(V @ Wt))]) @ Wt
    print(f"  {tag} MHC fit RMSD {np.sqrt((((P - pc) @ R - (Q - qc))**2).sum() / len(P)):.2f} A")
    tr = gemmi.Transform(gemmi.Mat33(R.T.tolist()), gemmi.Vec3(*qc))   # x -> R'(x-pc) + qc
    for model in st:
        for ch in model:
            for r in ch:
                for a in r:
                    a.pos = gemmi.Position(*tr.apply(a.pos - gemmi.Position(*pc)))
    st.make_mmcif_document().write_file(f"{out}/{pdb_id}_{tag}.cif")

native.make_mmcif_document().write_file(f"{out}/{pdb_id}_native.cif")
print(f"{cls} -> {out}")
