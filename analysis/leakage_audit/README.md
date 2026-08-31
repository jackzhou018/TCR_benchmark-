# Leakage audit — 126 scoreable targets vs the pre-cutoff record

Six-test audit of the benchmark against the 259 TCR-pMHC complexes released on or before
2021-09-30, plus the follow-up question of whether closeness to that record predicts accuracy.
Read-only with respect to the benchmark: nothing outside this directory was touched.

## Look at this first

- `figures/fig_identity_vs_dockq.png` — Global DockQ against four measures of closeness to the
  pre-cutoff record (concatenated TCR identity, TCR Vβ identity, mean all-chain identity, multimer
  TM-score), all four methods, Spearman ρ per method.
- `figures/fig_dockq_by_class.png` — accuracy by leakage class, and the near-duplicate minus clean
  lift with bootstrap intervals.
- `leakage_audit_report.html` — the full report: six tests, per-target table for all 126, method
  and limits. Open in a browser. Also published as an artifact.
- `results/audit_table.csv` — one row per target in the requested column order.

## Headline

| | |
|---|---|
| targets | 126 |
| failing the temporal cutoff (release ≤ 2021-09-30) | 0 (36 deposited before it, released after) |
| violating the 90/95% TCR rule | 42 |
| all chains >40% to one pre-cutoff entry | 43, of which 10 clear 70% on every chain |
| multimer TM > 0.90, complete + correct correspondence | 54 |
| clean of tests 2/3/5 | 42, every one with a partial prior |
| independent clusters after deduplication | 66 |

Applying every rule literally keeps 42 targets (25 after within-set deduplication). Excluding only
the 10 near-duplicates and the 32 TCR-redundant targets keeps 84 (46 independent).

Accuracy tracks leakage: median Global DockQ on the 10 near-duplicates versus the 42 clean targets
is +0.16 (AF3), +0.26 (Protenix), +0.23 (ESMFold2), +0.16 (TCRmodel2). ESMFold2 shows the effect
despite receiving no MSA and no template, so it is not template retrieval; and the strongest
correlate, TM-score to the closest pre-cutoff complex, is confounded with how canonical the target
is, so this cannot on its own separate memorisation from typicality.

## Reproduce

Scripts run in this order and write into `results/`. They expect the repository root at
`/14TBDrive/6TBDrive1_backup/benchmark_fresh` and a scratch directory for the 259 downloaded
mmCIFs (~364 MB) and their cropped PDBs (~52 MB), which are not kept here.

```
fetch_refs.py     259 pre-cutoff entries from RCSB          -> ref_cif/
crop_refs.py      crop to benchmark domains, benchmark      -> ref_pdb/   (213 of 259 croppable)
                  chain letters
dates.py          deposition + release dates (RCSB API)     -> dates.csv
seqaudit.py       21,549 target x reference chain-set       -> pairwise_target_vs_ref.csv
                  identities (global NW, BLOSUM62, -11/-1)     tcr_concat_vs_ref.csv
rules.py          tests 1-3 and partial priors a-c          -> rules.csv
within.py         test 6, within-benchmark clustering       -> within_pairs.csv, clusters.csv
binary_pmhc.py    partial prior d: same peptide in a         -> binary_pmhc.csv
                  pre-cutoff pMHC with no TCR
assemble.py       + Foldseek-Multimer report (test 5)       -> audit_full.csv
finalize.py       recommendations                            -> audit_final.csv, audit_table.csv
corr.py           identity vs Global DockQ                   -> identity_vs_dockq.csv
figs.py           the two figures                            -> figures/
rf_esm.py         side check: can the published RF run on ESMFold2 predictions (it cannot)
```

Foldseek step, run between `crop_refs.py` and `assemble.py`:

```
foldseek easy-multimersearch <natives> <ref_pdb> fs_aln fs_tmp --exact-tmscore 1 \
  -e 1000 --max-seqs 1000
```
using the binary in the `pepgym-curate` environment (10.941cd33).

## Caveats

- The reference set is a **proxy**. No predictor publishes its training list; this is TCR3d ∪
  STCRDab filtered to ≤ 2021-09-30, as assembled in `analysis/training_sequence_similarity/`.
- Test 5 does not discriminate on this problem: whole-complex TM-score has a median of 0.895 across
  the whole benchmark and correlates with peptide identity at ρ = 0.14. Corroboration only.
- Test 3's literal >40% threshold is nearly free to clear for a 9-mer peptide; the ≥70% subset is
  the one worth acting on.
- 213 of 259 references cropped cleanly, so test 5 covers 82% of the reference set. Tests 1-4 and 6
  use all 259.
- MSA-level leakage is untested, and the audit assumes the 2021-09-30 cutoff applies to every
  method — which is documented for AlphaFold 3 (server-enforced) and TCRmodel2 (launcher flag) but
  not, anywhere in this repository, for ESMFold2.
