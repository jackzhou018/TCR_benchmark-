"""Local Biohub ESMFold2 inference with pair-trunk taps and a diffusion trace.

Two instrumentation points, neither of which touches model mathematics:

* Pair trunk -- plain ``torch`` forward hooks on ``model.folding_trunk.blocks``.
  ``FoldingTrunk`` is an ordinary ``nn.ModuleList``, so no source patch is
  needed. A pre-hook on the trunk itself counts which refinement loop we are
  in; taps only fire on the requested loop (the final one by default).

* Diffusion sampler -- ``DiffusionStructureHead.sample`` is a long method with
  the state we want living in a local, so we take its *installed source*,
  insert one recording statement at a unique anchor, and exec it back into the
  module namespace. The arithmetic is therefore verbatim upstream code; if
  upstream ever changes so the anchor is missing, this raises instead of
  silently tracing the wrong thing.
"""
from __future__ import annotations

import argparse
import inspect
import json
import textwrap
import time
from pathlib import Path

import numpy as np
import torch

import common as C

MODEL_ID = "biohub/ESMFold2"
MODEL_REVISION = "8fc3ff471022fdce52c77030685eb775de0c00a3"
ESMC_ID = "biohub/ESMC-6B"
ESMC_REVISION = "45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a"
DIFF_TRACE: list = []

# Anchors are written against the *dedented* source (getsource on a method
# keeps its 4-space class indent, which textwrap.dedent then strips).
_ANCHOR = "\n        x_denoised_prev = x_denoised\n\n    result:"
_RECORD = (
    "\n        x_denoised_prev = x_denoised\n"
    "        _LPT_DIFF_TRACE.append({\n"
    "            'step': int(step_idx),\n"
    "            'sigma_tm': float(sigma_tm_val),\n"
    "            't_hat': float(t_hat_val),\n"
    "            'sigma_t': float(sigma_t_val),\n"
    "            'gamma': float(gamma.item()),\n"
    "            'x': x.float().cpu().numpy().copy(),\n"
    "            'x_denoised': x_denoised.float().cpu().numpy().copy(),\n"
    "        })\n\n    result:"
)


def patch_sampler():
    """Install the traced ``sample``; returns (original, source_sha256)."""
    from transformers.models.esmfold2 import modeling_esmfold2_common as M

    cls = M.DiffusionStructureHead
    original = cls.sample
    src = textwrap.dedent(inspect.getsource(original))
    import hashlib

    digest = hashlib.sha256(src.encode()).hexdigest()
    if _ANCHOR not in src:
        raise RuntimeError(
            "DiffusionStructureHead.sample no longer contains the trace anchor; "
            "refusing to patch (installed source sha256=%s)" % digest
        )
    ns = dict(M.__dict__)
    ns["_LPT_DIFF_TRACE"] = DIFF_TRACE
    exec(compile(src.replace(_ANCHOR, _RECORD), "<traced_sample>", "exec"), ns)
    cls.sample = ns["sample"]
    return original, digest


class TrunkTap:
    """Capture the pair state before block 1 and after every trunk block."""

    def __init__(self, model, sink, target_loop: int):
        self.trunk = model.folding_trunk
        self.sink = sink                 # sink(layer_idx, tensor[L,L,C])
        self.target_loop = target_loop
        self.loop = -1
        self.handles: list = []

    def __enter__(self):
        def on_trunk_enter(_m, _inp):
            self.loop += 1

        self.handles.append(self.trunk.register_forward_pre_hook(on_trunk_enter))

        def pre_block0(_m, inp):
            if self.loop == self.target_loop:
                self.sink(0, inp[0])

        self.handles.append(
            self.trunk.blocks[0].register_forward_pre_hook(pre_block0)
        )
        for k, blk in enumerate(self.trunk.blocks):
            def post(_m, _inp, out, k=k):
                if self.loop == self.target_loop:
                    self.sink(k + 1, out)
            self.handles.append(blk.register_forward_hook(post))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def build_input(pdb_id: str):
    from esm.models.esmfold2 import ESMFold2InputBuilder, ProteinInput
    from esm.models.esmfold2 import StructurePredictionInput

    seqs = C.read_fasta(pdb_id)
    chains = C.chains_for(pdb_id)
    spi = StructurePredictionInput(
        sequences=[ProteinInput(id=ch, sequence=seqs[ch]) for ch in chains]
    )
    return spi, chains


def load_model(device="cuda"):
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    model = ESMFold2Model.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=torch.float32,
        esmc_precision="bf16",
    )
    return model.to(device).eval()


class _NullTap:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def run_one(model, builder, pdb_id, *, seed, num_loops, num_steps, mode,
            out_dir: Path, per_category: int, trace_diffusion: bool,
            instrument: bool = True):
    """mode='pool' -> keep sampled pairs only; mode='full' -> dump every layer."""
    spi, chains = build_input(pdb_id)
    feats, chain_infos = builder.prepare_input(spi, seed=seed, device=model.device)

    fasta = C.read_fasta(pdb_id)
    lengths = [len(fasta[ch]) for ch in chains]
    n_res = sum(lengths)
    labels = np.concatenate([[ch] * len(fasta[ch]) for ch in chains])

    if mode == "pool":
        nxyz, _, present = C.native_ca_stack(pdb_id, chains=chains)
        pairs = C.sample_pairs(labels, per_category,
                               seed=C.stable_seed(pdb_id), present=present,
                               native_xyz=nxyz)
        ii = torch.as_tensor(pairs[:, 0], device=model.device)
        jj = torch.as_tensor(pairs[:, 1], device=model.device)
        store: dict[int, np.ndarray] = {}

        def sink(k, z):
            z = z[0] if z.dim() == 4 else z
            feat = (z[ii, jj] + z[jj, ii]).float()      # symmetrised, [K, C]
            store[k] = feat.cpu().numpy().astype(np.float16)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        store = {}

        def sink(k, z):
            z = z[0] if z.dim() == 4 else z
            zf = z.float()
            sym = (zf + zf.transpose(0, 1)).cpu().numpy()
            mx = float(np.abs(sym).max())
            if not np.isfinite(mx) or mx > 60000:
                raise RuntimeError(f"layer {k}: fp16 overflow risk (max {mx})")
            np.save(out_dir / f"layer_{k:03d}.npy", sym.astype(np.float16))
            store[k] = mx

    DIFF_TRACE.clear()
    torch.manual_seed(seed)
    t0 = time.time()
    tap = (TrunkTap(model, sink, target_loop=num_loops) if instrument
           else _NullTap())                          # loops are 0..num_loops
    with tap:
        out = model(
            **feats,
            num_loops=num_loops,
            num_diffusion_samples=1,
            num_sampling_steps=num_steps if trace_diffusion else None,
        )
    runtime = time.time() - t0

    meta = {
        "pdb_id": pdb_id,
        "chains": chains,
        "chain_lengths": dict(zip(chains, lengths)),
        "n_res": n_res,
        "n_layers_captured": len(store),
        "num_loops": num_loops,
        "total_trunk_passes": num_loops + 1,
        "num_sampling_steps": num_steps,
        "seed": seed,
        "runtime_s": round(runtime, 2),
        "mean_plddt": float(out["plddt"][0].mean()),
        "ptm": float(out["ptm"][0]) if "ptm" in out else None,
        "iptm": float(out["iptm"][0]) if "iptm" in out else None,
    }
    return out, feats, chain_infos, store, meta, labels, (
        pairs if mode == "pool" else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", nargs="+", required=True)
    ap.add_argument("--mode", choices=["pool", "full"], default="pool")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-loops", type=int, default=10)
    ap.add_argument("--num-steps", type=int, default=100)
    ap.add_argument("--per-category", type=int, default=500)
    ap.add_argument("--out", type=Path, default=C.CACHE / "esmfold2")
    ap.add_argument("--trace-diffusion", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--no-instrument", action="store_true",
                    help="stock model: no trunk taps, no sampler patch")
    args = ap.parse_args()
    if args.no_instrument:
        args.trace_diffusion = False

    src_sha = None
    if args.trace_diffusion:
        _, src_sha = patch_sampler()

    model = load_model()
    from esm.models.esmfold2 import ESMFold2InputBuilder

    builder = ESMFold2InputBuilder()
    args.out.mkdir(parents=True, exist_ok=True)

    for pdb_id in args.pdb:
        tag = args.out / pdb_id
        if args.skip_existing and (tag / "meta.json").exists():
            print(f"skip {pdb_id}")
            continue
        tag.mkdir(parents=True, exist_ok=True)
        out, feats, chain_infos, store, meta, labels, pairs = run_one(
            model, builder, pdb_id,
            seed=args.seed, num_loops=args.num_loops, num_steps=args.num_steps,
            mode=args.mode, out_dir=tag / "layers",
            per_category=args.per_category,
            trace_diffusion=args.trace_diffusion,
            instrument=not args.no_instrument,
        )
        meta["esmfold2_sample_src_sha256"] = src_sha

        if args.no_instrument:
            np.save(tag / "final_atom_coords.npy",
                    out["sample_atom_coords"][0].float().cpu().numpy())
        elif args.mode == "pool":
            ks = sorted(store)
            np.savez_compressed(
                tag / "pairs.npz",
                feats=np.stack([store[k] for k in ks]),      # [L+1, K, C]
                layers=np.array(ks),
            )
            xyz, nlab, present = C.native_ca_stack(pdb_id, chains=meta["chains"])
            assert (nlab == labels).all(), f"{pdb_id}: chain label mismatch"
            d = np.linalg.norm(xyz[pairs[:, 0]] - xyz[pairs[:, 1]], axis=-1)
            np.savez_compressed(
                tag / "labels.npz",
                pairs=pairs, dist=d, bins=C.digitize(d),
                cat=C.pair_category_ids(labels, pairs),
                near=(d < C.NEAR_CUT),
                natural_bin_prior=C.bin_prior(labels, xyz, present),
            )
        else:
            builder_res = builder.decode(out, feats, chain_infos,
                                         num_diffusion_samples=1,
                                         complex_id=pdb_id)
            (tag / "final_model.cif").write_text(builder_res.complex.to_mmcif())
            np.save(tag / "final_atom_coords.npy",
                    out["sample_atom_coords"][0].float().cpu().numpy())
            np.save(tag / "atom_mask.npy",
                    feats["atom_attention_mask"][0].cpu().numpy())
            if DIFF_TRACE:
                np.savez_compressed(
                    tag / "diffusion.npz",
                    x=np.stack([f["x"][0] for f in DIFF_TRACE]),
                    x_denoised=np.stack([f["x_denoised"][0] for f in DIFF_TRACE]),
                    step=np.array([f["step"] for f in DIFF_TRACE]),
                    sigma_tm=np.array([f["sigma_tm"] for f in DIFF_TRACE]),
                    t_hat=np.array([f["t_hat"] for f in DIFF_TRACE]),
                    sigma_t=np.array([f["sigma_t"] for f in DIFF_TRACE]),
                    gamma=np.array([f["gamma"] for f in DIFF_TRACE]),
                )
                meta["n_diffusion_frames"] = len(DIFF_TRACE)
        C.write_json(tag / "meta.json", meta)
        print(json.dumps(meta))


if __name__ == "__main__":
    main()
