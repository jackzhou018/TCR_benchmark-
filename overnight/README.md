# overnight — what AlphaFold 3's sampler settings actually cost

Does AF3 need 200 denoising steps and 5 diffusion samples to fold a TCR–pMHC
complex? Four tracks over the **same 30 complexes**, the **same MSAs**, the
**same seed**, on one idle RTX PRO 6000, run serially.

| track | steps | samples | GPU s / prediction | speed-up | Global DockQ | Δ vs baseline (95% CI) | paired p |
|---|---|---|---|---|---|---|---|
| `t1_baseline`  | 200 | 5 | **10.04** ± 0.03 | 1.00× | 0.7212 | — | — |
| `t2_fewsteps`  | 160 | 5 | **8.79** ± 0.03 | 1.14× | 0.7143 | −0.007 (−0.026, +0.007) | 0.43 |
| `t3_onesample` | 200 | 1 | **6.02** ± 0.01 | 1.67× | 0.7143 | −0.007 (−0.021, +0.006) | 0.34 |
| `t4_both`      | 160 | 1 | **5.57** ± 0.01 | **1.80×** | 0.7109 | −0.010 (−0.032, +0.007) | 0.33 |

**All three shortcuts are free.** Every confidence interval straddles zero;
no track loses a complex from the acceptable band (DockQ ≥ 0.23 in 30/30
everywhere). Both shortcuts together return the answer in 55% of the time.

Add featurisation (a constant 3.3 s of CPU, the same in every track) for the
end-to-end number: 13.4 s → 8.9 s per complex; 401 s → 267 s for all 30.

## 1. Where the diffusion converges — `figures/01_diffusion_convergence.png`

The traced pass (`t0_trace`) records AF3's coordinate state after every one of
the 200 production denoising steps, for all 5 samples of all 30 complexes —
30 000 frames. Mean Cα RMSD of frame *f* against the final frame of the same
sample:

| criterion | step |
|---|---|
| mean Cα RMSD to final < 2.0 Å | 150 |
| **mean Cα RMSD to final < 1.0 Å** | **158** |
| every complex under 1.0 Å | 159 |
| Cα RMSD to native within 0.1 Å of its plateau | 163 |
| mean Cα RMSD to final < 0.5 Å | 164 |

Four independent criteria land in 150–164, so **158 → 160** is the rounded step
count used by tracks 2 and 4.

The curve is the story: for the first ~140 steps the RMSD to the final frame
sits at ≈0.84 σ — the state *is* noise at the scale the schedule sets, and no
structure has settled. AF3's schedule is
`noise_schedule(linspace(0, 1, steps+1))` with p=7, which holds σ above 5 Å
until ~70% of the way through. Accuracy against the native plateaus at
3.37 Å around step 163 and does not improve afterwards.

**Caveat, and why track 2 exists.** Running with `steps=160` is *not* the
200-step trajectory stopped at 158: AF3 re-discretises the same σ range into
160 coarser steps. The convergence analysis picks the number; track 2 is the
experiment that tests it. It holds (−0.007 DockQ, p=0.43).

## 2. Why the speed-up stops at 1.8× — `figures/02_time_by_track.png`

Fitting `t = trunk + c·(steps × samples)` across all 120 timed predictions
gives **trunk = 4.86 s fixed** and **5.08 ms per sample-step** (R²=0.995).
Half of a default AF3 run is the Pairformer trunk and its 10 recycles, which
neither shortcut touches. 5×200 sample-steps = 5.08 s of diffusion; drive that
to zero and 1.9× is the ceiling.

## 3. Accuracy — `figures/03_accuracy_by_track.png`, `figures/04_time_vs_accuracy.png`

Full-atom DockQ v2 against `input_data/natives/`, explicit identity `--mapping`
(per `../CLAUDE.md`), plus Cα RMSD / TM-score / interface RMSD on the
natively-resolved residues. Per-complex differences are what the middle panels
show — the across-complex spread is how hard each complex is, which every
track shares.

Dropping to 1 sample costs the *ranking*: AF3 normally returns the best of 5
by ranking score. On this sample that selection is worth −0.007 DockQ on
average, i.e. nothing detectable at n=30.

## Method

**Same 30 complexes, every track.** `scripts/pick_sample.py`, seed 20260827,
drawn from the benchmark's 126 scoreable complexes (READY, native built, not in
`excluded.csv`) that also have an AlphaFold Server zip. 28 class I, 2 class II,
397–416 residues.

**Same MSAs.** `inputs/<PDB>.json` carries the paired and unpaired `.a3m` MSAs
and the template hits **verbatim out of the AlphaFold Server result zips the
benchmark's own AF3 predictions came from**, built by
`layer_probe_trajectories/scripts/af3_inputs.py` *unchanged*. No MSA search was
run and the AF3 data/download pipeline was not touched.
`tests/test_overnight.py` byte-compares every chain's MSA against its zip.

**Timing is real.** All 30 complexes are 397–416 tokens, so AF3's stock bucket
ladder pads every one to 512 and JIT-compiles once per track. That compile —
and the second, tokamax's kernel autotune — is charged to two warm-up
predictions run and discarded before the clock starts. `run_inference()` ends
with a device→host `tree_map`, so the wall time around it is not async.
Standard deviations of 0.01–0.03 s over 30 predictions are the evidence this
worked.

**Instrumentation is not in the timing.** `t0_trace` applies only
`af3_patch`'s `diffusion_head.sample` rewrite (the trajectory is the scan's
already-computed, discarded `y`), never the Pairformer layer-stack patch.
It costs ~2%, so it is a separate pass and `t1_baseline` is stock AF3.
Parity: mean |ΔDockQ| between the two = 0.0008, max 0.0122.

## Reproduce

```bash
python scripts/pick_sample.py                              # sample.csv (frozen seed)
python ../layer_probe_trajectories/scripts/af3_inputs.py \
       --pdb $(tail -n+2 sample.csv | cut -d, -f1) --out inputs
bash   scripts/run_all.sh early                            # t0_trace, t1, t3   (~25 min)
~/anaconda3/envs/local_esmfold2/bin/python scripts/traj_rmsd.py    # -> 160
bash   scripts/run_all.sh late 160                         # t2, t4             (~15 min)
~/anaconda3/envs/dockq/bin/python          scripts/score.py
~/anaconda3/envs/local_esmfold2/bin/python scripts/figures.py
~/anaconda3/envs/local_esmfold2/bin/python -m pytest tests -q
```

## Layout

```
sample.csv        the 30 complexes
inputs/           AF3 input JSON per complex: AlphaFold Server MSAs + templates + seed
runs/<track>/     per complex: model.cif, meta.json (+ ca_traj.npz in t0_trace)
results/          <track>_times.csv, accuracy.csv, traj_rmsd_*.csv,
                  convergence.json, summary.csv, paired_deltas.csv
figures/          01 convergence, 02 time, 03 accuracy, 04 trade-off
logs/             raw stdout/stderr of every pass
tests/            35 self-checks (sample, MSA identity, track coverage, seeds)
```

## Scope

n=30, one GPU, one complex class (TCR–pMHC, ~400 residues, all in the 512
bucket), one seed per complex. The 1.8× and the "no accuracy cost" both apply
to *this* regime. Larger complexes shift the trunk/diffusion split — the trunk
is O(N²)-ish in the pair track while diffusion scales with atoms — so the
ceiling moves. A different sample would move the DockQ deltas within their CIs.
