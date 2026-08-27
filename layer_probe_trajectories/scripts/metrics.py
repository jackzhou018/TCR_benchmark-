"""Structure-comparison metrics shared by the layer-probe and diffusion paths.

Everything here takes Ca coordinate arrays in the benchmark chain order, so a
frame from a probe reconstruction and a frame from a diffusion trajectory are
scored by identical code.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

import common as C

DOCKQ_BIN = "/home/yash/anaconda3/envs/dockq/bin/DockQ"


def kabsch(P: np.ndarray, Q: np.ndarray):
    """Rotation+translation taking P onto Q (both [N,3])."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, qc - R @ pc


def apply_rt(X, R, t):
    return X @ R.T + t


def superpose(P, Q, on=None):
    """Return P superposed onto Q, fitting on `on` (index array) if given."""
    sel = slice(None) if on is None else on
    R, t = kabsch(P[sel], Q[sel])
    return apply_rt(P, R, t)


def rmsd(P, Q):
    return float(np.sqrt(((P - Q) ** 2).sum(-1).mean()))


def rmsd_after_fit(P, Q, on=None):
    return rmsd(superpose(P, Q, on)[on if on is not None else slice(None)],
                Q[on if on is not None else slice(None)])


def _d0(L: int) -> float:
    return max(1.24 * (L - 15) ** (1 / 3) - 1.8, 0.5) if L > 15 else 0.5


def tm_score(P: np.ndarray, Q: np.ndarray) -> float:
    """TM-score of P against reference Q with a 1:1 residue correspondence.

    Standard iterative-extension search: seed on fragments of decreasing
    length, superimpose, keep residues under a cutoff, repeat to convergence,
    and take the best score seen.
    """
    L = len(Q)
    d0 = _d0(L)
    best = 0.0
    for frag in [L, L // 2, L // 4, L // 8, 4]:
        frag = max(int(frag), 4)
        for start in range(0, L - frag + 1, max(frag // 2, 1)):
            sel = np.arange(start, start + frag)
            for _ in range(20):
                moved = superpose(P, Q, sel)
                d = np.linalg.norm(moved - Q, axis=-1)
                score = float((1.0 / (1.0 + (d / d0) ** 2)).mean())
                best = max(best, score)
                cutoff = max(d0, 3.0)
                new = np.flatnonzero(d < cutoff)
                while len(new) < 4:
                    cutoff += 0.5
                    new = np.flatnonzero(d < cutoff)
                if len(new) == len(sel) and (new == sel).all():
                    break
                sel = new
    return best


def group_masks(labels: np.ndarray):
    pmhc, tcr = C.groups_for(labels)
    return np.isin(labels, pmhc), np.isin(labels, tcr)


def native_interface_residues(native: np.ndarray, labels: np.ndarray,
                              cutoff: float = 10.0) -> np.ndarray:
    """Indices of residues within `cutoff` of the other side of the pMHC/TCR
    interface, defined on the NATIVE (used only for scoring, never for fitting)."""
    p, t = group_masks(labels)
    D = np.linalg.norm(native[:, None] - native[None, :], axis=-1)
    sub = D[np.ix_(p, t)]
    pi = np.flatnonzero(p)[(sub < cutoff).any(1)]
    ti = np.flatnonzero(t)[(sub < cutoff).any(0)]
    return np.sort(np.concatenate([pi, ti]))


def interface_rmsd(pred, native, labels, cutoff=10.0) -> float:
    sel = native_interface_residues(native, labels, cutoff)
    return rmsd_after_fit(pred, native, sel)


def peptide_rmsd_after_mhc_align(pred, native, labels) -> float:
    pmhc, _ = C.groups_for(labels)
    mhc = np.flatnonzero(np.isin(labels, [c for c in pmhc if c != "B"]))
    pep = np.flatnonzero(labels == "B")
    moved = superpose(pred, native, mhc)
    return rmsd(moved[pep], native[pep])


def contact_pr(pred, native, labels, cutoff=8.0):
    """Precision / recall of pMHC-TCR Ca contacts under `cutoff` A."""
    p, t = group_masks(labels)
    idx = np.ix_(np.flatnonzero(p), np.flatnonzero(t))
    nat = np.linalg.norm(native[:, None] - native[None, :], axis=-1)[idx] < cutoff
    prd = np.linalg.norm(pred[:, None] - pred[None, :], axis=-1)[idx] < cutoff
    tp = int((nat & prd).sum())
    prec = tp / max(int(prd.sum()), 1)
    rec = tp / max(int(nat.sum()), 1)
    return prec, rec, int(nat.sum()), int(prd.sum())


# ---------------------------------------------------------------------------
# Grouped DockQ: pMHC (A+B) as receptor vs TCR (C+D) as ligand.
# DockQ's --mapping is chain-to-chain, so the grouping is expressed by writing
# a re-chained copy (A+B -> R, C+D -> L) with contiguous residue numbering.
# This is a genuine two-body interface, not a four-chain complex forced into a
# two-chain call.
# ---------------------------------------------------------------------------
def _write_grouped_pdb(path: Path, xyz, labels):
    lines, serial = [], 1
    for grp, chains in zip(("R", "L"), C.groups_for(labels)):
        resi = 1
        for k in np.flatnonzero(np.isin(labels, chains)):
            x, y, z = xyz[k]
            lines.append(
                f"ATOM  {serial:5d}  CA  ALA {grp}{resi:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
            )
            serial += 1
            resi += 1
        lines.append("TER")
    path.write_text("\n".join(lines) + "\nEND\n")


def dockq_grouped(pred, native, labels) -> dict:
    """DockQ for the (A+B) vs (C+D) interface. Returns {} if DockQ fails."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        _write_grouped_pdb(td / "model.pdb", pred, labels)
        _write_grouped_pdb(td / "native.pdb", native, labels)
        try:
            r = subprocess.run(
                [DOCKQ_BIN, str(td / "model.pdb"), str(td / "native.pdb"),
                 "--mapping", "RL:RL", "--json", str(td / "out.json"),
                 "--allowed_mismatches", "0"],
                capture_output=True, text=True, timeout=300,
            )
            import json

            if not (td / "out.json").exists():
                return {"dockq_error": (r.stderr or r.stdout)[-200:]}
            d = json.loads((td / "out.json").read_text())
            best = d.get("best_result", {})
            iface = next(iter(best.values())) if best else {}
            return {
                "dockq": iface.get("DockQ"),
                "dockq_fnat": iface.get("fnat"),
                "dockq_irms": iface.get("iRMSD"),
                "dockq_lrms": iface.get("LRMSD"),
                "global_dockq": d.get("GlobalDockQ"),
            }
        except Exception as e:                                  # noqa: BLE001
            return {"dockq_error": str(e)[:200]}


def score_frame(pred, native, labels) -> dict:
    """All per-frame structural metrics. `pred` is not modified."""
    fitted = superpose(pred, native)
    prec, rec, n_nat, n_prd = contact_pr(fitted, native, labels)
    out = {
        "ca_rmsd": rmsd(fitted, native),
        "tm_score": tm_score(pred, native),
        "interface_rmsd": interface_rmsd(pred, native, labels),
        "peptide_rmsd_mhc_aligned": peptide_rmsd_after_mhc_align(
            pred, native, labels),
        "contact_precision": prec,
        "contact_recall": rec,
        "n_native_contacts": n_nat,
        "n_pred_contacts": n_prd,
    }
    out.update(dockq_grouped(fitted, native, labels))
    return out
