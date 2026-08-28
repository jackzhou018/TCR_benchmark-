# pDockQ2

pDockQ2 (pDockQ_i) — Zhu, Shenoy, Kundrotas & Elofsson, *Bioinformatics* **39**(7):btad424 (2023),
<https://doi.org/10.1093/bioinformatics/btad424>. Same benchmark shape as `../run_pdockq.py`.

```
python pDockQ/pdockq2/run_pdockq2.py [--check N]   # results/pdockq2_all.csv, pdockq2_per_structure.csv
python pDockQ/pdockq2/plot_pdockq2.py              # results/pdockq2_vs_dockq_AF3.png
```

`src/pdockq2.py` is the authors' script, verbatim from
<https://gitlab.com/ElofssonLab/afm-benchmark/-/raw/main/src/pdockq2.py>.

**AF3 only.** pDockQ2 replaces pDockQ's contact count with the PAE, and only AF3 kept a PAE
matrix — pulled from the server zips (`models/AF3/AF3/fold_<id>_truncated_tcr_pmhc.zip`,
`full_data_0.json`, whose `model_0.cif` is byte-identical to the scored prediction). ESMFold2
and Protenix outputs on disk are CIF + summary confidences only, so they would have to be
re-run to be scored.

**Per chain, not per chain pair.** pDockQ2 is defined for one chain against everything it
touches, so rows are chains (519) where pDockQ v1's rows are chain pairs; per-structure is the
mean over chains. Same 126 scoreable complexes, `review_status` carried through, 9J4S excluded.

`run_pdockq2.py` vectorises the reference's O(N²) Python loops; `--check N` re-runs N whole
structures through `src/pdockq2.py` as a subprocess and asserts every chain agrees (0
disagreements on the first 2 structures / 8 chains).

AF3, n=126, vs Global DockQ: pDockQ ρ=0.60, **pDockQ2 ρ=0.65**, ipTM ρ=0.65. pDockQ2 spreads
over the full 0–1 range (median 0.67) where pDockQ v1 is compressed near 0.2 (median 0.18).
