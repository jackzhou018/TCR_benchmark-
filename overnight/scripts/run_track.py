#!/usr/bin/env python
"""One timed AlphaFold 3 pass over the 30-structure sample.

    env -u LD_LIBRARY_PATH ~/anaconda3/envs/af3/bin/python run_track.py \
        --track t1_baseline --steps 200 --samples 5

A "track" is one (diffusion steps, diffusion samples) setting run over every
structure in ../sample.csv, in manifest order, on one GPU, serially. Inputs are
the AlphaFold Server MSAs/templates (../inputs/<PDB>.json, built unchanged by
layer_probe_trajectories/scripts/af3_inputs.py) and the server's own model seed,
so the only thing that differs between tracks is the sampler setting.

Timing: AF3 pads to a bucket (all 30 complexes are 397-416 tokens -> the 512
bucket) and JIT-compiles once per bucket+config. That compile is charged to a
warm-up prediction that is run and thrown away before the timed loop, so no
structure's number carries it. run_inference() ends with a tree_map to numpy,
which blocks on the device, so the wall time around it is real.

--trace additionally patches diffusion_head.sample to return its per-step
coordinate state (the scan's discarded `y`). That costs an extra host transfer,
so a traced pass is NOT used for the timing comparison.
"""
from __future__ import annotations

import argparse, csv, json, os, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LPT = ROOT / "layer_probe_trajectories" / "scripts"
AF3_SRC = Path("/data/alphafold3")
MODEL_DIR = Path("/data/af3_models")

sys.path.insert(0, str(LPT))
import common as C                                              # noqa: E402


def sample_ids() -> list[str]:
    with open(ROOT / "overnight" / "sample.csv") as f:
        return [r["pdb_id"] for r in csv.DictReader(f)]


def patch_diffusion_only():
    """Apply ONLY af3_patch's diffusion-trajectory rewrite.

    af3_patch.apply_patches() also switches the 48-block Pairformer to
    layer_stack(with_per_layer_inputs=True), which materialises a
    [48, N, N, 128] tensor -- irrelevant here and a large slowdown.
    """
    import af3_patch
    from alphafold3.model.network import diffusion_head

    diffusion_head.sample = af3_patch._rewrite(
        diffusion_head.sample,
        [(af3_patch._DIFF_OLD, af3_patch._DIFF_NEW),
         (af3_patch._DIFF_OLD_RET, af3_patch._DIFF_NEW_RET)],
        diffusion_head, name="sample",
    )
    return dict(af3_patch.PATCHES)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--samples", type=int, required=True)
    ap.add_argument("--recycles", type=int, default=10)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--flash", default="triton")
    ap.add_argument("--pdb", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--redo", action="store_true")
    # AF3's own default bucket ladder (run_alphafold.py _BUCKETS). Every complex
    # in the sample is 397-416 tokens, so all 30 land in the 512 bucket and the
    # JIT/autotune happens exactly once per track -- in the warm-up. Without it
    # featurisation pads to the exact token count and each structure recompiles.
    ap.add_argument("--buckets", default="256,512,768,1024,1280,1536,2048,2560,"
                                         "3072,3584,4096,4608,5120")
    args = ap.parse_args()

    pdbs = args.pdb or sample_ids()
    if args.limit:
        pdbs = pdbs[: args.limit]

    sys.argv = sys.argv[:1]           # tokamax feeds sys.argv to absl on first use
    sys.path.insert(0, str(AF3_SRC))
    patches = patch_diffusion_only() if args.trace else {}

    import jax
    import run_alphafold as ra
    from alphafold3.common import folding_input
    from alphafold3.constants import chemical_components
    from alphafold3.data import featurisation
    # ponytail: the af3 env has no pandas; diffusion_traj only needs it inside
    # main(), and we import it for cif_atoms/ca_indices. Stub instead of
    # installing into the AF3 env, which would risk its pinned numpy/jax.
    import types
    sys.modules.setdefault("pandas", types.ModuleType("pandas"))
    import diffusion_traj as DT

    cfg = ra.make_model_config(
        flash_attention_implementation=args.flash,
        num_diffusion_samples=args.samples,
        num_recycles=args.recycles,
    )
    cfg.heads.diffusion.eval.steps = args.steps
    runner = ra.ModelRunner(config=cfg, device=jax.local_devices(backend="gpu")[0],
                            model_dir=MODEL_DIR)

    out_root = ROOT / "overnight" / "runs" / args.track
    out_root.mkdir(parents=True, exist_ok=True)
    times_csv = ROOT / "overnight" / "results" / f"{args.track}_times.csv"
    times_csv.parent.mkdir(parents=True, exist_ok=True)

    ccd = chemical_components.Ccd()
    buckets = tuple(int(b) for b in args.buckets.split(","))

    def featurise(pdb_id):
        fi = folding_input.Input.from_json(
            (ROOT / "overnight" / "inputs" / f"{pdb_id}.json").read_text())
        ex = featurisation.featurise_input(fold_input=fi, buckets=buckets, ccd=ccd,
                                           verbose=False)[0]
        return fi, ex

    # ---- warm-up: pay the parameter load and the 512-bucket JIT here -------
    t0 = time.time()
    _ = runner.model_params
    load_s = time.time() - t0
    _fi, _ex = featurise(pdbs[0])
    # Twice: tokamax autotunes its Pallas kernels on the first execution and JAX
    # then re-traces with the tuned configs, so pass 1 alone leaves a second
    # (still inflated) run behind. Two warm-ups put every compile before the clock.
    warm = []
    for _ in range(2):
        t0 = time.time()
        runner.run_inference(_ex, jax.random.PRNGKey(int(_fi.rng_seeds[0])))
        warm.append(time.time() - t0)
    print(f"[warmup] params {load_s:.1f}s  compile+run "
          f"{' '.join(f'{w:.1f}s' for w in warm)}", flush=True)
    del _ex, _fi

    fields = ["track", "pdb_id", "n_residues", "padded_tokens", "steps", "samples",
              "recycles", "seed", "featurise_s", "inference_s", "total_s",
              "ranking_score", "iptm", "ptm", "traced"]
    have = {}
    if times_csv.exists() and not args.redo:
        with open(times_csv) as f:
            have = {r["pdb_id"]: r for r in csv.DictReader(f)}
    rows = []

    for k, pdb_id in enumerate(pdbs, 1):
        tag = out_root / pdb_id
        if pdb_id in have and (tag / "meta.json").exists() and not args.redo:
            rows.append(have[pdb_id])
            print(f"[{k}/{len(pdbs)}] {pdb_id} skip (done)", flush=True)
            continue
        tag.mkdir(parents=True, exist_ok=True)
        chains = C.chains_for(pdb_id)
        fasta = C.read_fasta(pdb_id)
        n_res = sum(len(fasta[c]) for c in chains)

        t0 = time.time()
        fi, ex = featurise(pdb_id)
        feat_s = time.time() - t0

        seed = int(fi.rng_seeds[0])
        t0 = time.time()
        result = runner.run_inference(ex, jax.random.PRNGKey(seed))
        inf_s = time.time() - t0

        infs = runner.extract_inference_results(batch=ex, result=result,
                                                target_name=pdb_id)
        # AlphaFold Server ranks its 5 samples and calls the best model_0; match that.
        rank = [float(np.asarray(r.metadata["ranking_score"]).reshape(-1)[0]) for r in infs]
        best = int(np.argmax(rank))
        (tag / "model.cif").write_text(infs[best].predicted_structure.to_mmcif())

        meta = dict(pdb_id=pdb_id, track=args.track, chains=chains, n_res=n_res,
                    seed=seed, steps=args.steps, samples=args.samples,
                    recycles=args.recycles, sample_index=best,
                    ranking_scores=rank, traced=bool(args.trace),
                    padded_tokens=int(np.asarray(ex["seq_mask"]).size),
                    featurise_s=round(feat_s, 3), inference_s=round(inf_s, 3),
                    af3_identifier=result["__identifier__"].decode(errors="replace"),
                    patch_source_sha256=patches)
        for key in ("ptm", "iptm", "fraction_disordered", "ranking_score"):
            if key in infs[best].metadata:
                meta[key] = float(np.asarray(infs[best].metadata[key]).reshape(-1)[0])

        if args.trace:
            recs, _ = DT.cif_atoms(tag / "model.cif")
            cai = DT.ca_indices(recs, chains)
            mask = np.asarray(result["diffusion_samples"]["mask"])[best].reshape(-1).astype(bool)
            traj = np.asarray(result["diffusion_samples"]["lpt_trajectory"])   # [F,S,T,A,3]
            ca = traj.reshape(traj.shape[0], traj.shape[1], -1, 3)[:, :, mask][:, :, cai]
            final = np.asarray(
                result["diffusion_samples"]["atom_positions"]).reshape(
                    traj.shape[1], -1, 3)[:, mask][:, cai]
            dev = float(np.abs(ca[-1] - final).max())
            if dev > 1e-3:
                raise SystemExit(f"{pdb_id}: last frame != model output ({dev:g})")
            np.savez_compressed(tag / "ca_traj.npz", ca=ca.astype(np.float32),
                                noise_levels=np.asarray(
                                    result["diffusion_samples"]["lpt_noise_levels"]),
                                best=best)
            meta["n_frames"] = int(ca.shape[0])
            meta["last_frame_dev"] = dev

        (tag / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        rows.append(dict(track=args.track, pdb_id=pdb_id, n_residues=n_res,
                         padded_tokens=meta["padded_tokens"], steps=args.steps,
                         samples=args.samples, recycles=args.recycles, seed=seed,
                         featurise_s=round(feat_s, 3), inference_s=round(inf_s, 3),
                         total_s=round(feat_s + inf_s, 3),
                         ranking_score=round(rank[best], 4),
                         iptm=round(meta.get("iptm", float("nan")), 4),
                         ptm=round(meta.get("ptm", float("nan")), 4),
                         traced=int(bool(args.trace))))
        print(f"[{k}/{len(pdbs)}] {pdb_id} feat={feat_s:6.1f}s infer={inf_s:6.1f}s "
              f"rank={rank[best]:.3f}", flush=True)

        with open(times_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fields)
            w.writeheader()
            w.writerows(rows)
        del result

    print(f"-> {times_csv}")


if __name__ == "__main__":
    main()
