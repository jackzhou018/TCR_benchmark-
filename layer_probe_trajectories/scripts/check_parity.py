"""Instrumented vs stock: same seed must give the same final structure.

The taps and the sampler patches are supposed to be observational. This runs
the target twice per model -- once instrumented, once stock -- and reports the
deviation of the final all-atom prediction. Small nonzero deviations are
expected on GPU (non-deterministic reduction order in fused kernels), so the
check is a tolerance, not bit-equality; the tolerance is far below any
structurally meaningful difference.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

import common as C

ESM_PY = "/home/yash/anaconda3/envs/local_esmfold2/bin/python"
AF3_PY = "/home/yash/anaconda3/envs/af3/bin/python"
HERE = Path(__file__).resolve().parent


def run(cmd, env_clean=False):
    full = (["env", "-u", "LD_LIBRARY_PATH"] if env_clean else []) + cmd
    r = subprocess.run(full, cwd=HERE, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-3000:], file=sys.stderr)
        print(r.stderr[-3000:], file=sys.stderr)
        raise SystemExit(f"failed: {' '.join(cmd[:6])}")
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["esmfold2", "af3"])
    ap.add_argument("--pdb", default=C.TARGET)
    ap.add_argument("--out", type=Path, default=C.CACHE / "parity")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="max allowed per-atom deviation, angstrom")
    ap.add_argument("--num-loops", type=int, default=10)
    ap.add_argument("--num-steps", type=int, default=100)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    a, b = args.out / f"{args.model}_instr", args.out / f"{args.model}_stock"

    if args.model == "esmfold2":
        base = [ESM_PY, "esmfold2_runner.py", "--pdb", args.pdb,
                "--mode", "full", "--num-loops", str(args.num_loops),
                "--num-steps", str(args.num_steps)]
        run(base + ["--trace-diffusion", "--out", str(a)])
        run(base + ["--no-instrument", "--out", str(b)])
    else:
        base = [AF3_PY, "af3_runner.py", "--pdb", args.pdb, "--mode", "full"]
        run(base + ["--out", str(a)], env_clean=True)
        run(base + ["--no-patch", "--out", str(b)], env_clean=True)

    fa = "final_atom_coords.npy" if args.model == "esmfold2" \
        else "final_atom_positions.npy"
    xa = np.load(a / args.pdb / fa).reshape(-1, 3)
    xb = np.load(b / args.pdb / fa).reshape(-1, 3)
    if xa.shape != xb.shape:
        raise SystemExit(f"shape mismatch {xa.shape} vs {xb.shape}")
    finite = np.isfinite(xa).all(1) & np.isfinite(xb).all(1)
    d = np.linalg.norm(xa[finite] - xb[finite], axis=-1)
    print(f"{args.model}: atoms={int(finite.sum())} "
          f"max|d|={d.max():.6f} A  mean|d|={d.mean():.6f} A  "
          f"rmsd={np.sqrt((d ** 2).mean()):.6f} A")
    ok = d.max() <= args.tol
    print("PARITY OK" if ok else f"PARITY FAIL (tol {args.tol} A)")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
