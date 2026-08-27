#!/usr/bin/env python
"""Per-diffusion-step Ca RMSD from the traced pass, and where it converges.

    ~/anaconda3/envs/local_esmfold2/bin/python traj_rmsd.py

Reads runs/t0_trace/<PDB>/ca_traj.npz -- Ca coordinates after every production
denoising step of AF3's own sampler, all 5 diffusion samples kept -- and writes

  results/traj_rmsd_per_structure.csv   mean over samples, per structure, per step
  results/traj_rmsd_mean.csv            mean over structures, per step
  results/convergence.json              the step counts the criteria pick
  figures/01_diffusion_convergence.png

Two curves, both Kabsch-superposed before the RMSD:

  to_final    frame f vs the LAST frame of the SAME sample. This is what
              "converged" means for a sampler: the point past which more steps
              stop moving the coordinates. It is the criterion used to choose
              the reduced step count.
  to_native   frame f vs the benchmark native. Accuracy, not convergence --
              it plateaus at the model's error, which is not zero.

Caveat carried into the README: AF3 builds its noise schedule as
noise_schedule(linspace(0, 1, steps+1)), so running with steps=S is a COARSER
walk over the same sigma range, not a truncation of the 200-step walk at f=S.
The convergence step is therefore a principled starting guess; track 2 is what
actually tests it.
"""
from __future__ import annotations

import csv, json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "layer_probe_trajectories" / "scripts"))
import common as C            # noqa: E402
import metrics as M           # noqa: E402

TRACE = ROOT / "overnight" / "runs" / "t0_trace"
RES = ROOT / "overnight" / "results"
FIG = ROOT / "overnight" / "figures"
# "same structure" thresholds on the Ca RMSD to the final frame, in angstrom.
THRESHOLDS = (2.0, 1.0, 0.5, 0.25)


def per_structure(pdb_id: str):
    d = np.load(TRACE / pdb_id / "ca_traj.npz")
    ca = d["ca"].astype(np.float64)                      # [F, S, N, 3]
    F, S = ca.shape[0], ca.shape[1]
    native, labels, present = C.native_ca_stack(pdb_id, chains=C.chains_for(pdb_id))
    if ca.shape[2] != len(native):
        raise SystemExit(f"{pdb_id}: {ca.shape[2]} traj CA != {len(native)} native")
    nat, sel = native[present], np.flatnonzero(present)

    to_final = np.empty((F, S))
    to_native = np.empty((F, S))
    step_delta = np.full((F, S), np.nan)
    for s in range(S):
        ref = ca[-1, s]
        for f in range(F):
            x = ca[f, s]
            to_final[f, s] = M.rmsd(M.superpose(x, ref), ref)
            to_native[f, s] = M.rmsd(M.superpose(x[sel], nat), nat)
            if f:
                prev = ca[f - 1, s]
                step_delta[f, s] = M.rmsd(M.superpose(x, prev), prev)
    return dict(pdb_id=pdb_id, n_frames=F, n_samples=S,
                to_final=to_final, to_native=to_native, step_delta=step_delta,
                sigma=d["noise_levels"][1:])


def main() -> None:
    pdbs = [r["pdb_id"] for r in csv.DictReader(
        open(ROOT / "overnight" / "sample.csv"))]
    have = [p for p in pdbs if (TRACE / p / "ca_traj.npz").exists()]
    if len(have) != len(pdbs):
        print(f"WARNING: {len(have)}/{len(pdbs)} traced", file=sys.stderr)
    out = [per_structure(p) for p in have]
    F = out[0]["n_frames"]
    if any(o["n_frames"] != F for o in out):
        raise SystemExit("traced structures disagree on frame count")

    RES.mkdir(parents=True, exist_ok=True)
    with open(RES / "traj_rmsd_per_structure.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pdb_id", "step", "sigma", "n_samples",
                    "ca_rmsd_to_final_mean", "ca_rmsd_to_final_sd",
                    "ca_rmsd_to_native_mean", "ca_rmsd_to_native_sd",
                    "ca_rmsd_step_delta_mean"])
        for o in out:
            for f in range(F):
                w.writerow([o["pdb_id"], f + 1, f"{o['sigma'][f]:.6g}", o["n_samples"],
                            f"{o['to_final'][f].mean():.4f}", f"{o['to_final'][f].std():.4f}",
                            f"{o['to_native'][f].mean():.4f}", f"{o['to_native'][f].std():.4f}",
                            "" if f == 0 else f"{np.nanmean(o['step_delta'][f]):.4f}"])

    fin = np.stack([o["to_final"].mean(1) for o in out])      # [P, F]
    nat = np.stack([o["to_native"].mean(1) for o in out])
    with np.errstate(invalid="ignore"):
        import warnings
        warnings.filterwarnings("ignore", "Mean of empty slice")
        dlt = np.stack([np.nanmean(o["step_delta"], 1) for o in out])
    mean_fin, mean_nat = fin.mean(0), nat.mean(0)
    steps = np.arange(1, F + 1)

    with open(RES / "traj_rmsd_mean.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "sigma", "n_structures",
                    "ca_rmsd_to_final_mean", "ca_rmsd_to_final_median",
                    "ca_rmsd_to_final_p90", "ca_rmsd_to_native_mean",
                    "ca_rmsd_step_delta_mean"])
        for f in range(F):
            w.writerow([f + 1, f"{out[0]['sigma'][f]:.6g}", len(out),
                        f"{mean_fin[f]:.4f}", f"{np.median(fin[:, f]):.4f}",
                        f"{np.percentile(fin[:, f], 90):.4f}",
                        f"{mean_nat[f]:.4f}",
                        "" if f == 0 else f"{np.nanmean(dlt[:, f]):.4f}"])

    def first_below(curve, thr):
        """First step from which the curve stays below thr (not just touches it)."""
        ok = curve <= thr
        run = np.flatnonzero(~ok)
        start = (run[-1] + 1) if len(run) else 0
        return int(start + 1) if start < len(curve) else None

    crit = {f"mean_to_final<{t}A": first_below(mean_fin, t) for t in THRESHOLDS}
    # per-structure agreement: the step by which EVERY structure is under 1 A
    crit["all_structures_to_final<1.0A"] = first_below(fin.max(0), 1.0)
    # accuracy plateau: to-native within 0.1 A of its final value
    crit["to_native_within_0.1A_of_plateau"] = first_below(
        np.abs(mean_nat - mean_nat[-1]), 0.1)
    primary = crit["mean_to_final<1.0A"]
    rounded = int(np.ceil(primary / 10.0) * 10) if primary else None
    payload = dict(n_structures=len(out), n_samples=int(out[0]["n_samples"]),
                   total_steps=F, criteria=crit,
                   primary_criterion="mean_to_final<1.0A",
                   converged_step=primary, rounded_step=rounded,
                   mean_to_final_at_converged=float(mean_fin[primary - 1]) if primary else None,
                   mean_to_native_final=float(mean_nat[-1]))
    (RES / "convergence.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))

    # ---------------- figure ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
    # -- A: the whole trajectory. Log y over 6 decades, because for most of the
    #    run the state IS noise: the RMSD to the final frame sits at ~0.84 sigma.
    for p in range(fin.shape[0]):
        ax[0].plot(steps, fin[p], color="0.85", lw=0.6, zorder=1)
    ax[0].plot(steps, out[0]["sigma"], color="0.35", lw=1.2, ls="-.", zorder=2,
               label=r"noise level $\sigma$")
    ax[0].plot(steps, mean_fin, color="#1f77b4", lw=2.2, zorder=4,
               label=f"mean of {len(out)} complexes")
    ax[0].fill_between(steps, np.percentile(fin, 10, 0), np.percentile(fin, 90, 0),
                       color="#1f77b4", alpha=0.2, zorder=3, label="10-90th pct")
    ax[0].axhline(1.0, color="#d62728", ls=":", lw=1)
    if primary:
        ax[0].axvline(primary, color="#d62728", ls="--", lw=1.4)
        ax[0].annotate(f"1 $\\rm\\AA$ at step {primary}\nused: {rounded}",
                       (primary - 6, 40), fontsize=9, color="#d62728", ha="right")
    ax[0].set_yscale("log")
    ax[0].set_ylim(1e-3, 1e4)          # the very last frame is RMSD 0 to itself
    ax[0].set_xlim(0, F)
    ax[0].set_xlabel("diffusion step (of 200)")
    ax[0].set_ylabel(r"C$\alpha$ RMSD to final frame ($\rm\AA$)")
    ax[0].set_title("The sampler is noise until $\\sigma$ is")
    ax[0].legend(frameon=False, fontsize=8, loc="lower left")

    # -- B: the last 80 steps, linear, where the coordinates are a structure.
    z = steps >= 120
    ax[1].plot(steps[z], mean_nat[z], color="#2ca02c", lw=2.2,
               label="to native (accuracy)")
    ax[1].fill_between(steps[z], np.percentile(nat, 10, 0)[z],
                       np.percentile(nat, 90, 0)[z], color="#2ca02c", alpha=0.18)
    ax[1].plot(steps[z], mean_fin[z], color="#1f77b4", lw=2.0, ls="--",
               label="to final frame (convergence)")
    ax[1].axhline(mean_nat[-1], color="#2ca02c", ls=":", lw=1)
    ax[1].annotate(f"model error {mean_nat[-1]:.2f} $\\rm\\AA$", (199, mean_nat[-1]),
                   ha="right", va="bottom", fontsize=8, color="#2ca02c")
    for t in (1.0, 0.5):
        st = crit[f"mean_to_final<{t}A"]
        if st:
            ax[1].annotate(f"{t} $\\rm\\AA$ @ {st}", (st, t), fontsize=8,
                           color="#d62728", ha="left", va="bottom")
            ax[1].plot([st], [t], "o", ms=4, color="#d62728")
    if primary:
        ax[1].axvline(primary, color="#d62728", ls="--", lw=1.4)
    ax[1].set_ylim(0, 8)
    ax[1].set_xlim(120, F)
    ax[1].set_xlabel("diffusion step (of 200)")
    ax[1].set_ylabel(r"mean C$\alpha$ RMSD ($\rm\AA$)")
    ax[1].set_title("Last 80 steps: accuracy plateaus, then coordinates freeze")
    ax[1].legend(frameon=False, fontsize=8)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"AF3 diffusion trajectory, {len(out)} TCR-pMHC complexes "
                 f"x {out[0]['n_samples']} samples, 200 steps", y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "01_diffusion_convergence.png", dpi=200, bbox_inches="tight")
    print(f"-> {FIG/'01_diffusion_convergence.png'}")


if __name__ == "__main__":
    main()
