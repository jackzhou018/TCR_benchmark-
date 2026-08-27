"""Local AlphaFold 3 inference with Pairformer taps and a diffusion trace.

Run with a cleaned LD_LIBRARY_PATH (the system CUDA 12.8 libs shadow the pip
nvidia wheels and JAX silently falls back to CPU):

    env -u LD_LIBRARY_PATH .../envs/af3/bin/python af3_runner.py ...
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import common as C

MODEL_DIR = Path("/data/af3_models")
AF3_SRC = Path("/data/alphafold3")


def load_fold_input(json_path: Path):
    from alphafold3.common import folding_input

    return folding_input.Input.from_json(json_path.read_text())


def build_runner(*, num_recycles, num_samples, num_steps, flash, device_idx=0):
    import jax

    import run_alphafold as ra

    cfg = ra.make_model_config(
        flash_attention_implementation=flash,
        num_diffusion_samples=num_samples,
        num_recycles=num_recycles,
        return_embeddings=False,
        return_distogram=True,
    )
    if num_steps is not None:
        cfg.heads.diffusion.eval.steps = num_steps
    dev = jax.local_devices(backend="gpu")[device_idx]
    return ra.ModelRunner(config=cfg, device=dev, model_dir=MODEL_DIR), cfg, ra


def token_layout(batch_dict, n_res_expected, chains, lengths):
    """Real-token indices in benchmark chain order + per-token chain letter."""
    mask = np.asarray(batch_dict["seq_mask"]).astype(bool)
    asym = np.asarray(batch_dict["asym_id"])[mask]
    idx = np.flatnonzero(mask)
    if len(idx) != n_res_expected:
        raise ValueError(f"{len(idx)} real tokens != {n_res_expected} residues")
    order = list(dict.fromkeys(asym.tolist()))
    if len(order) != len(chains):
        raise ValueError(f"{len(order)} asym ids != {len(chains)} chains")
    labels = []
    for a, ch, n in zip(order, chains, lengths):
        k = int((asym == a).sum())
        if k != n:
            raise ValueError(f"chain {ch}: {k} tokens != {n} residues")
        labels += [ch] * k
    return idx, np.array(labels)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", nargs="+", required=True)
    ap.add_argument("--inputs", type=Path, default=C.CACHE / "af3_inputs")
    ap.add_argument("--out", type=Path, default=C.CACHE / "af3")
    ap.add_argument("--mode", choices=["pool", "full"], default="pool")
    ap.add_argument("--num-recycles", type=int, default=10)
    ap.add_argument("--num-samples", type=int, default=1)
    ap.add_argument("--num-steps", type=int, default=None)
    ap.add_argument("--per-category", type=int, default=500)
    ap.add_argument("--bucket", type=int, default=None,
                    help="single bucket size; default = AF3's bucket ladder")
    ap.add_argument("--flash", default="triton")
    ap.add_argument("--no-patch", action="store_true",
                    help="run stock AF3 (for the instrumented/stock parity test)")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    import sys

    # tokamax (AF3's flash-attention backend) lazily runs absl's flag parser
    # over sys.argv the first time a config value is read, and dies on our
    # argparse flags. Our own parsing is done, so hide them.
    sys.argv = sys.argv[:1]
    sys.path.insert(0, str(AF3_SRC))
    patches = {}
    if not args.no_patch:
        import af3_patch

        patches = af3_patch.apply_patches()

    import jax
    from alphafold3.constants import chemical_components
    from alphafold3.data import featurisation

    runner, cfg, ra = build_runner(
        num_recycles=args.num_recycles, num_samples=args.num_samples,
        num_steps=args.num_steps, flash=args.flash,
    )
    buckets = [args.bucket] if args.bucket else None
    args.out.mkdir(parents=True, exist_ok=True)

    for pdb_id in args.pdb:
        tag = args.out / pdb_id
        if args.skip_existing and (tag / "meta.json").exists():
            print(f"skip {pdb_id}", flush=True)
            continue
        tag.mkdir(parents=True, exist_ok=True)
        fold_input = load_fold_input(args.inputs / f"{pdb_id}.json")
        fasta = C.read_fasta(pdb_id)
        chains = C.chains_for(pdb_id)
        lengths = [len(fasta[ch]) for ch in chains]
        n_res = sum(lengths)

        ccd = chemical_components.Ccd(user_ccd=fold_input.user_ccd)
        examples = featurisation.featurise_input(
            fold_input=fold_input, buckets=buckets, ccd=ccd, verbose=False
        )
        example = examples[0]
        seed = fold_input.rng_seeds[0]
        t0 = time.time()
        result = runner.run_inference(example, jax.random.PRNGKey(seed))
        runtime = time.time() - t0

        idx, labels = token_layout(example, n_res, chains, lengths)
        infs = runner.extract_inference_results(
            batch=example, result=result, target_name=pdb_id
        )
        # AlphaFold Server returns 5 samples ranked by ranking_score and calls the
        # top one model_0, which is what the benchmark scored. extract_inference_results
        # hands them back in sample order, so pick the ranked best explicitly.
        best = int(np.argmax([float(np.asarray(r.metadata["ranking_score"]).reshape(-1)[0])
                              for r in infs]))
        inf = infs[best]

        meta = {
            "pdb_id": pdb_id, "chains": chains,
            "chain_lengths": dict(zip(chains, lengths)), "n_res": n_res,
            "seed": int(seed), "num_recycles": args.num_recycles,
            "num_diffusion_samples": args.num_samples,
            "sample_index": best,
            "diffusion_steps": int(cfg.heads.diffusion.eval.steps),
            "num_pairformer_layers": int(cfg.evoformer.pairformer.num_layer),
            "pair_channel": int(cfg.evoformer.pair_channel),
            "padded_tokens": int(np.asarray(example["seq_mask"]).size),
            "flash_attention": args.flash,
            "runtime_s": round(runtime, 2),
            "patched": not args.no_patch,
            "patch_source_sha256": patches,
            "af3_identifier": result["__identifier__"].decode(errors="replace"),
        }

        if not args.no_patch:
            pre = np.asarray(result["lpt_pair_pre"], dtype=np.float32)
            layers = np.asarray(result["lpt_pair_layers"], dtype=np.float32)
            pre = pre[np.ix_(idx, idx)]
            layers = layers[:, idx][:, :, idx]
            meta["n_layers_captured"] = int(layers.shape[0]) + 1

            if args.mode == "pool":
                xyz, nlab, present = C.native_ca_stack(pdb_id, chains=chains)
                assert (nlab == labels).all()
                pairs = C.sample_pairs(labels, args.per_category,
                                       seed=C.stable_seed(pdb_id),
                                       present=present, native_xyz=xyz)
                i, j = pairs[:, 0], pairs[:, 1]
                stack = [pre[i, j] + pre[j, i]]
                stack += [layers[k][i, j] + layers[k][j, i]
                          for k in range(layers.shape[0])]
                np.savez_compressed(
                    tag / "pairs.npz",
                    feats=np.stack(stack).astype(np.float16),
                    layers=np.arange(len(stack)),
                )
                d = np.linalg.norm(xyz[i] - xyz[j], axis=-1)
                np.savez_compressed(
                    tag / "labels.npz", pairs=pairs, dist=d,
                    bins=C.digitize(d),
                    cat=C.pair_category_ids(labels, pairs),
                    near=(d < C.NEAR_CUT),
                    natural_bin_prior=C.bin_prior(labels, xyz, present),
                )
            else:
                ld = tag / "layers"
                ld.mkdir(exist_ok=True)
                np.save(ld / "layer_000.npy",
                        (pre + pre.transpose(1, 0, 2)).astype(np.float16))
                for k in range(layers.shape[0]):
                    z = layers[k]
                    np.save(ld / f"layer_{k + 1:03d}.npy",
                            (z + z.transpose(1, 0, 2)).astype(np.float16))
                traj = np.asarray(result["diffusion_samples"]["lpt_trajectory"])
                np.savez_compressed(
                    tag / "diffusion.npz",
                    x=traj[:, best],
                    noise_levels=np.asarray(
                        result["diffusion_samples"]["lpt_noise_levels"]),
                    atom_mask=np.asarray(
                        result["diffusion_samples"]["mask"])[best],
                    final=np.asarray(
                        result["diffusion_samples"]["atom_positions"])[best],
                    token_idx=idx,
                )
                meta["n_diffusion_frames"] = int(traj.shape[0])

        (tag / "final_model.cif").write_text(inf.predicted_structure.to_mmcif())
        np.save(tag / "final_atom_positions.npy",
                np.asarray(result["diffusion_samples"]["atom_positions"])[best])
        np.save(tag / "token_idx.npy", idx)
        (tag / "token_labels.txt").write_text("".join(labels))
        for k in ("ptm", "iptm", "fraction_disordered", "ranking_score"):
            if k in inf.metadata:
                meta[k] = float(np.asarray(inf.metadata[k]).reshape(-1)[0])
        C.write_json(tag / "meta.json", meta)
        print(json.dumps(meta)[:600], flush=True)


if __name__ == "__main__":
    main()
