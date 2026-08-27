"""Export a model's denoising trajectory as an all-atom XTC + topology + CSV.

This is a SAMPLING trajectory, not a layer probe: frame f is the coordinate
state after production denoising step f of the official sampler. No frames are
interpolated or invented, and the last frame is asserted to reproduce the
model's ordinary output.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import common as C
import metrics as M

AA3 = {v: k for k, v in C.THREE_TO_ONE.items() if len(k) == 3}


def cif_atoms(path: Path):
    """(records, xyz) for every ATOM/HETATM row, in file order."""
    hdr, rows = [], []
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if s.startswith("_atom_site."):
            hdr.append(s.split(".", 1)[1])
        elif s.startswith(("ATOM ", "HETATM")):
            rows.append(s.split())
    c = {n: i for i, n in enumerate(hdr)}
    ch_key = "auth_asym_id" if "auth_asym_id" in c else "label_asym_id"
    seq_key = "auth_seq_id" if "auth_seq_id" in c else "label_seq_id"
    recs, xyz = [], []
    for r in rows:
        recs.append({
            "name": r[c["label_atom_id"]].strip('"'),
            "resname": r[c["label_comp_id"]],
            "chain": r[c[ch_key]],
            "resid": int(r[c[seq_key]]),
            "element": r[c["type_symbol"]],
        })
        xyz.append([float(r[c["Cartn_x"]]), float(r[c["Cartn_y"]]),
                    float(r[c["Cartn_z"]])])
    return recs, np.asarray(xyz)


def write_xtc(topology: Path, frames: np.ndarray, out: Path) -> None:
    """frames [F,N,3] in angstrom; MDAnalysis converts to nm on XTC write."""
    import MDAnalysis as mda

    u = mda.Universe(str(topology))
    if len(u.atoms) != frames.shape[1]:
        raise ValueError(f"topology {len(u.atoms)} atoms != {frames.shape[1]}")
    with mda.Writer(str(out), n_atoms=len(u.atoms)) as w:
        for f in frames:
            u.atoms.positions = f.astype(np.float32)
            w.write(u.atoms)


def write_pdb(path: Path, recs, xyz=None) -> None:
    """Topology PDB holding the FINAL frame's coordinates.

    They must be a real structure: ChimeraX infers connectivity from distance,
    so an all-origin topology loads as one atomic heap with no backbone and
    shows nothing once the XTC is attached. Frame 0 is no better -- it is the
    sampler's initial noise, a ~1000 A cloud that also overflows the PDB's
    8-column coordinate field. The last frame is the model's own prediction.
    """
    if xyz is None:
        xyz = np.zeros((len(recs), 3))
    lines = []
    last_chain = None
    for i, a in enumerate(recs, start=1):
        if last_chain is not None and a["chain"] != last_chain:
            lines.append("TER")
        last_chain = a["chain"]
        nm = a["name"]
        nm = f" {nm:<3s}" if len(nm) < 4 else nm
        x, y, z = xyz[i - 1]
        lines.append(
            f"ATOM  {i:5d} {nm}{a['resname']:>4s} {a['chain']}{a['resid']:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          "
            f"{a['element']:>2s}"
        )
    lines.append("TER")
    path.write_text("\n".join(lines) + "\nEND\n")


def ca_indices(recs, chains) -> np.ndarray:
    """Indices of Ca atoms, ordered by benchmark chain then residue."""
    out = []
    for ch in chains:
        sel = [(a["resid"], i) for i, a in enumerate(recs)
               if a["chain"] == ch and a["name"] == "CA"]
        out += [i for _, i in sorted(sel)]
    return np.asarray(out)


def load_esmfold2(cache: Path):
    d = np.load(cache / "diffusion.npz")
    mask = np.load(cache / "atom_mask.npy").astype(bool)
    frames = d["x"][:, mask]
    final = np.load(cache / "final_atom_coords.npy")[mask]
    meta = pd.DataFrame({
        "step": d["step"], "sigma_prev": d["sigma_tm"], "t_hat": d["t_hat"],
        "sigma_next": d["sigma_t"], "gamma": d["gamma"],
    })
    return frames, final, meta


def load_af3(cache: Path):
    d = np.load(cache / "diffusion.npz")
    mask = d["atom_mask"].astype(bool)             # [tokens, max_atoms]
    frames = d["x"].reshape(d["x"].shape[0], -1, 3)[:, mask.reshape(-1)]
    final = d["final"].reshape(-1, 3)[mask.reshape(-1)]
    nl = d["noise_levels"]
    meta = pd.DataFrame({
        "step": np.arange(len(frames)),
        "sigma_prev": nl[:-1], "sigma_next": nl[1:],
    })
    return frames, final, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["esmfold2", "af3"])
    ap.add_argument("--pdb", default=C.TARGET)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    recs, cif_xyz = cif_atoms(args.cache / "final_model.cif")
    frames, final, meta = (load_esmfold2 if args.model == "esmfold2"
                           else load_af3)(args.cache)
    if frames.shape[1] != len(recs):
        raise SystemExit(
            f"{frames.shape[1]} trajectory atoms != {len(recs)} topology atoms")

    # The stored final frame must BE the model's ordinary prediction.
    dev_final = float(np.abs(frames[-1] - final).max())
    dev_cif = float(np.abs(final - cif_xyz).max())
    if dev_final > args.tol:
        raise SystemExit(f"last frame deviates from model output by {dev_final}")
    print(f"last frame vs model output: max|d| = {dev_final:.2e}")
    print(f"model output vs written CIF: max|d| = {dev_cif:.2e}")

    fasta = C.read_fasta(args.pdb)
    chains = C.chains_for(args.pdb)
    cai = ca_indices(recs, chains)
    native, labels, present = C.native_ca_stack(args.pdb, chains=chains)
    if not present.all():
        raise SystemExit(f"{args.pdb}: native has unresolved residues")
    if len(cai) != len(native):
        raise SystemExit(f"{len(cai)} CA atoms != {len(native)} native CA")

    # Align every frame to the final state (visualisation only).
    ref = frames[-1]
    aligned = np.stack([M.superpose(f, ref) for f in frames])

    topo = args.out / f"{args.model}_{args.pdb}_diffusion_topology.pdb"
    write_pdb(topo, recs, aligned[-1])
    write_xtc(topo, aligned,
              args.out / f"{args.model}_{args.pdb}_diffusion.xtc")

    rows = []
    for k in range(len(frames)):
        s = M.score_frame(frames[k][cai], native, labels)
        s.update({
            "frame": k, "model": args.model, "pdb_id": args.pdb,
            "representation": "diffusion_sample_state",
            "module": ("structure_head.sample" if args.model == "esmfold2"
                       else "diffusion_head.sample"),
            "total_steps": len(frames),
            "normalized_progress": k / (len(frames) - 1),
            "n_atoms": int(frames.shape[1]),
            "aligned_to": "final denoising frame (Kabsch)",
            "notes": ("first production denoising step" if k == 0 else
                      ("final denoising step; equals model output"
                       if k == len(frames) - 1 else "production denoising step")),
        })
        rows.append(s)
    df = pd.concat([meta.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    front = ["frame", "step", "model", "pdb_id", "representation", "module",
             "total_steps", "normalized_progress", "sigma_prev", "sigma_next",
             "ca_rmsd", "tm_score", "interface_rmsd",
             "peptide_rmsd_mhc_aligned", "contact_precision",
             "contact_recall", "dockq"]
    df = df[[c for c in front if c in df] +
            [c for c in df.columns if c not in front]]
    df.to_csv(args.out / f"{args.model}_{args.pdb}_diffusion_frames.csv",
              index=False)
    print(df[["frame", "sigma_prev", "ca_rmsd", "tm_score",
              "interface_rmsd"]].to_string(index=False))


if __name__ == "__main__":
    main()
