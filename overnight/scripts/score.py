#!/usr/bin/env python
"""Accuracy of every track's predictions against the benchmark natives.

    ~/anaconda3/envs/dockq/bin/python score.py            # all tracks found
    ~/anaconda3/envs/dockq/bin/python score.py t1_baseline

Two families of number, both against input_data/natives/<PDB>.pdb:

  full-atom DockQ, via ../../scripts/run_dockq.py's own helpers, with an
  explicit identity --mapping (see ../../CLAUDE.md: without it DockQ optimizes
  the chain assignment and can swap the homologous TCRa/TCRb). `global_dockq`
  is DockQ v2's GlobalDockQ and is the headline accuracy number, as in the rest
  of the benchmark.

  Ca-only geometry, via layer_probe_trajectories/scripts/metrics.py, restricted
  to natively-resolved residues: global Ca RMSD, TM-score, interface RMSD and
  the peptide RMSD after MHC alignment.

-> results/accuracy.csv (one row per track x complex)
"""
from __future__ import annotations

import csv, os, sys, tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "layer_probe_trajectories" / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))
import common as C            # noqa: E402
import metrics as M           # noqa: E402
import run_dockq as RD        # noqa: E402

RUNS = ROOT / "overnight" / "runs"
RES = ROOT / "overnight" / "results"
ROLES = RD.ROLES


def ca_metrics(pdb_id, cif):
    chains = C.chains_for(pdb_id)
    native, labels, present = C.native_ca_stack(pdb_id, chains=chains)
    pred = C.read_cif_ca(cif, chains=chains)
    if len(pred) != len(native):
        raise ValueError(f"{pdb_id}: {len(pred)} predicted CA != {len(native)}")
    p, n, lab = pred[present], native[present], labels[present]
    fitted = M.superpose(p, n)
    prec, rec, n_nat, n_prd = M.contact_pr(fitted, n, lab)
    return dict(
        ca_rmsd=round(M.rmsd(fitted, n), 4),
        tm_score=round(M.tm_score(p, n), 4),
        interface_rmsd=round(M.interface_rmsd(p, n, lab), 4),
        peptide_rmsd_mhc_aligned=round(M.peptide_rmsd_after_mhc_align(p, n, lab), 4),
        contact_precision=round(prec, 4), contact_recall=round(rec, 4),
        n_native_contacts=n_nat, n_resolved=int(present.sum()), n_fasta=len(native),
    )


def main() -> None:
    want = sys.argv[1:] or sorted(d.name for d in RUNS.iterdir() if d.is_dir())
    meta = RD.native_meta()
    sample = list(csv.DictReader(open(ROOT / "overnight" / "sample.csv")))
    rows, bad = [], []

    for track in want:
        for k, s in enumerate(sample, 1):
            pdb = s["pdb_id"]
            cif = RUNS / track / pdb / "model.cif"
            if not cif.exists():
                bad.append((track, pdb, "no prediction"))
                continue
            with tempfile.TemporaryDirectory() as tmp:
                mp = f"{tmp}/model.pdb"
                RD.cif_to_pdb(str(cif), mp)
                try:
                    res = RD.score(mp, f"{ROOT}/input_data/natives/{pdb}.pdb")
                except Exception as e:                          # noqa: BLE001
                    bad.append((track, pdb, f"DockQ: {str(e)[:80]}"))
                    continue
            m = meta.get(pdb, {})
            roles = ROLES.get(m.get("mhc_class"), {})
            ifaces = res["best_result"]
            row = dict(track=track, pdb_id=pdb, mhc_class=m.get("mhc_class", ""),
                       resolution=m.get("resolution", ""),
                       global_dockq=round(res["GlobalDockQ"], 4),
                       n_interfaces=len(ifaces))
            for v in ifaces.values():
                c1, c2 = v["chain1"], v["chain2"]
                row[f"dockq_{roles.get(c1, c1)}-{roles.get(c2, c2)}"] = round(v["DockQ"], 4)
            row.update(ca_metrics(pdb, cif))
            rows.append(row)
            print(f"[{track} {k}/{len(sample)}] {pdb} GlobalDockQ={row['global_dockq']:.3f} "
                  f"CaRMSD={row['ca_rmsd']:.2f}", flush=True)

    cols = list(dict.fromkeys(c for r in rows for c in r))
    RES.mkdir(parents=True, exist_ok=True)
    with open(RES / "accuracy.csv", "w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(rows)
    print(f"-> {RES/'accuracy.csv'}  ({len(rows)} rows)")
    for b in bad:
        print("  MISSING", *b)


if __name__ == "__main__":
    main()
