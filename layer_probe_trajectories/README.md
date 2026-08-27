# Diffusion trajectories for TCR–pMHC complexes

Records the **denoising (sampling) trajectory** of **AlphaFold 3** and **Biohub
ESMFold2** on a benchmark complex: the coordinate state after every production
denoising step, exported as an all-atom XTC + topology PDB + a per-frame metric
CSV.

This is the model's *own* sampler, patched only to keep the coordinates it
already computes. Nothing is interpolated, and `diffusion_traj.py` asserts that
the last frame reproduces the model's ordinary prediction before writing
anything.

> Only the diffusion path is committed here. The layer-probe analysis that
> shares this directory is unfinished and is documented separately in
> `README.layerprobe.md` (not tracked).

## Requirements

Two conda envs, both on this host (`environment.yml` records the exact pins):

| env | used for |
|---|---|
| `local_esmfold2` | ESMFold2 inference, XTC/CSV export, metrics |
| `af3` | AlphaFold 3 inference only (writes `.npz` the other env reads) |
| `dockq` | DockQ v2, called as a subprocess by `metrics.py` |

Also needed, from the benchmark root: `input_data/truncated_structures/`
(FASTAs + manifests) and `input_data/natives/` — the runners read sequences from
there and the metrics score against the native. Chain letters are the
benchmark's, used verbatim (see `../CLAUDE.md`).

**AF3 must run with `LD_LIBRARY_PATH` unset.** A system CUDA install shadows the
pip nvidia wheels and JAX silently falls back to CPU. `run_diffusion.sh` already
does this.

## Usage

One command per complex, from `layer_probe_trajectories/`:

```bash
bash scripts/run_diffusion.sh 8RYO           # both models
bash scripts/run_diffusion.sh 8RYO af3       # one model
```

It runs the instrumented inference when `cache/<model>_target/<PDB>/` is missing
(ESMFold2 ~40 s, AF3 ~110 s incl. JAX compile) and then exports the trajectory.
Re-running is cheap — the inference is skipped once the cache exists. Override
the interpreters with `ESM=` / `AF3=` env vars.

Targets in `input_data/natives/excluded.csv` (currently 9J4S — no scoreable
interface) are refused.

### Output

```
outputs/<PDB>/<model>/
  <model>_<PDB>_diffusion.xtc              one frame per denoising step
  <model>_<PDB>_diffusion_topology.pdb     all-atom topology, final-frame coords
  <model>_<PDB>_diffusion_frames.csv       per-frame noise level + metrics
```

The CSV carries, per frame: the sampler's `sigma_prev`/`sigma_next`,
`normalized_progress`, and `ca_rmsd`, `tm_score`, `interface_rmsd`,
`peptide_rmsd_mhc_aligned`, contact precision/recall and DockQ against the
native.

The topology PDB holds the *final* frame's coordinates on purpose: ChimeraX
infers connectivity by distance, so an all-zero (or frame-0 noise) topology
loads as an atomic heap with no backbone.

### Running the stages by hand

```bash
ESM=/home/yash/anaconda3/envs/local_esmfold2/bin/python
AF3=/home/yash/anaconda3/envs/af3/bin/python

# ESMFold2: inference with the sampler traced
$ESM scripts/esmfold2_runner.py --pdb 8RYO --mode full --num-loops 10 \
     --num-steps 100 --trace-diffusion --out cache/esmfold2_target

# AF3: build the input JSON from the benchmark's AF-Server inputs, then run
$ESM scripts/af3_inputs.py --pdb 8RYO --out cache/af3_inputs
env -u LD_LIBRARY_PATH $AF3 scripts/af3_runner.py --pdb 8RYO --mode full \
     --num-recycles 10 --num-samples 5 --bucket <n_residues> --out cache/af3_target

# export either one
$ESM scripts/diffusion_traj.py --model af3 --pdb 8RYO \
     --cache cache/af3_target/8RYO --out outputs/8RYO/af3
```

### Sanity check

The patches are supposed to be observational — instrumented and stock runs of
the same seed must give the same structure:

```bash
$ESM scripts/check_parity.py --model esmfold2 --pdb 8RYO
$ESM scripts/check_parity.py --model af3      --pdb 8RYO
```

GPU reductions are non-deterministic, so this is a tolerance (0.05 Å per atom by
default), not bit-equality.

## Scripts

| file | role |
|---|---|
| `run_diffusion.sh` | the entry point: inference (if needed) → trajectory, per model |
| `esmfold2_runner.py` | ESMFold2 inference; `--trace-diffusion` records the sampler |
| `af3_inputs.py` | builds AF3 input JSON from the benchmark's AF-Server inputs |
| `af3_runner.py` | AF3 inference, applying `af3_patch.py` |
| `af3_patch.py` | rewrites `diffusion_head.sample` (and the Evoformer/Model wiring) from installed source at unique anchors |
| `diffusion_traj.py` | `.npz` → XTC + topology + per-frame metric CSV |
| `common.py` | paths, chain convention, FASTA/native handling |
| `metrics.py` | RMSD / TM-score / interface metrics / DockQ, shared by every frame |
| `check_parity.py` | instrumented vs stock, same seed |

## How the samplers are instrumented

**ESMFold2** keeps its sampler state in a local, so
`DiffusionStructureHead.sample` is patched by taking its *installed source*,
inserting one recording statement at a unique anchor, and exec'ing it back into
the module. A missing anchor raises rather than instrumenting the wrong thing.

**AlphaFold 3**'s `diffusion_head.sample` already computes the per-step
positions inside its scan and discards them into `_`; the patch keeps them. No
arithmetic is changed in either model.

## Caveats

1. **DockQ here is Cα-only**, to stay comparable with the layer-probe series
   that shares `metrics.py`. Fnat is a Cα-contact fraction — these numbers are
   not interchangeable with the benchmark's all-atom `output/DockQ/`.
2. **Grouped DockQ**: the pMHC(A+B)-vs-TCR(C+D) split is expressed by writing a
   re-chained copy (A+B → `R`, C+D → `L`) and calling DockQ with
   `--mapping RL:RL`.
3. **Frames are aligned to the final frame** (Kabsch, visualisation only). The
   metrics are computed on the unaligned coordinates.
4. **ESMFold2 ran without MSAs**, from the local `biohub/ESMFold2` checkpoint —
   not the API model `esmfold2-fast-2026-05` the benchmark scored. Its
   DockQ/RMSD are not interchangeable with `models/ESMFold2/results/`.
5. **AF3 uses the benchmark's AlphaFold Server MSAs and templates**
   (`models/AF3/AF3/*.zip`), not a re-run local data pipeline, so the inputs
   match the benchmark run.

## Viewing

```bash
vmd outputs/8RYO/af3/af3_8RYO_diffusion_topology.pdb \
    outputs/8RYO/af3/af3_8RYO_diffusion.xtc

pymol -d 'load outputs/8RYO/af3/af3_8RYO_diffusion_topology.pdb, traj; \
          load_traj outputs/8RYO/af3/af3_8RYO_diffusion.xtc, traj; \
          show ribbon; color skyblue, chain A; color yellow, chain B; \
          color salmon, chain C; color palegreen, chain D'
```

Frame 0 is the sampler's initial noise (~1000 Å cloud); the last frame is the
model's prediction.
