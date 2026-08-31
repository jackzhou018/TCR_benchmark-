# Cases for the divergence figure

Four where the two models disagree, five where they agree and are both right.
Each folder holds `<PDB>_native.cif`, `<PDB>_AF3.cif`, `<PDB>_ESMFold2.cif`. Both predictions
are rigid-body moved onto the native's MHC (Cα, class I chain A) by `../export_case.py`, so the
three files open in one frame and every difference you see is peptide or TCR. All four are
class I, `native_qc = ok`, `READY`.

Chains: A = MHC α1α2, B = peptide, C = TCRα V, D = TCRβ V.

Numbers are from `../divergence_with_metadata.csv`. "rot / com" = that model's TCR against the
native's after superposing MHC on MHC. "height" = TCR centre projected on the MHC-centre →
peptide-centre axis; native class I is never below +14 Å.

|  | 8RYO | 9NW2 | 9RU5 | 9GV7 |
|---|---|---|---|---|
| pMHC | HLA-A\*03 / ELFSYLIEK | H2-Db / ASNENMETM | HLA-A\*01 / FTSDYYQLY | HLA-A\*02 / SLAGGLDDMKA |
| AF3 rot / com / height | 162° / 53 Å / **−24 Å** | 132° / 5 Å / +29 Å | 12° / 4 Å / +29 Å | 160° / 20 Å / +31 Å |
| ESM rot / com / height | 3° / 0.1 Å / +28 Å | 17° / 2 Å / +29 Å | 167° / 12 Å / +27 Å | 116° / 12 Å / +29 Å |
| DockQ AF3 / ESM | 0.28 / **0.91** | 0.33 / 0.56 | **0.61** / 0.31 | 0.32 / 0.29 |
| ipTM AF3 / ESM | 0.59 / 0.75 | 0.52 / 0.50 | 0.56 / 0.43 | 0.55 / 0.47 |
| cross-model RMSD | 9.26 Å | 5.30 Å | 6.62 Å | 2.92 Å |

- **8RYO — wrong face.** AF3 docks the TCR on the β-sheet floor *under* the platform, the
  surface the domain truncation exposes; zero Cα contacts to the peptide. ESMFold2 is
  essentially the native. The largest class I disagreement in the set.
- **9NW2 — reversed polarity, same site.** AF3's TCR still straddles the groove (centre 5 Å
  from ESMFold2's, both +29 Å above the platform) but is turned 132° around on it. The other
  failure mode, and the one a DockQ number alone does not distinguish from 8RYO.
- **9RU5 — same flip, other model.** ESMFold2 is the one 167° off; AF3 is right. Rules out
  "AF3 is always the wrong one" (across the 20 largest disagreements: AF3 12, ESM 4, both 4).
- **9GV7 — agreement is not correctness.** The two predictions are the closest of the four
  (2.92 Å, 44° apart) and *both* are 116–160° from the native. An 11-mer bulged peptide is also
  the only one of the four where the models disagree about the peptide itself (1.25 Å).

In all four the fold is not the problem: TCR internal RMSD 0.49–1.73 Å, peptide conformation
0.19–1.25 Å. Neither model's ipTM separates its own failures (0.52–0.59 for AF3 whether it is
right or wrong).

## Both models right

The other end of the spectrum: cross-model RMSD ≤ 0.7 Å *and* both models accurate. Ranked by
how little the answer could have been copied — `complex_n_candidates` is the number of
pre-cutoff complexes similar enough to serve as a template, and `peptide_n_hits` the number of
pre-cutoff hits for the peptide itself.

| | 8RYP | 43RH | 8QFY | 8WUL | 8PJG |
|---|---|---|---|---|---|
| pMHC | HLA-A\*03 / ELFSYLIEK | H2-Q9 (mouse) / HALNVVHDW | HLA-E\*01 / RLPAKAPLL | HLA-A\*11 / VVGAVGVGK | HLA-DRB1 / PKYVKQNTLKLAR |
| class | I | I | I | I | **II** |
| DockQ AF3 / ESM | 0.86 / 0.81 | 0.85 / 0.77 | 0.94 / 0.92 | 0.76 / 0.74 | 0.945 / 0.945 |
| rot from native AF3 / ESM | 6° / 5° | 5° / 10° | 4° / 1° | 9° / 11° | 2° / 4° |
| cross-model RMSD | 0.69 Å | 0.55 Å | 0.21 Å | 0.45 Å | 0.20 Å |
| TCR identity to pre-cutoff | 97% | **64%** | 94% | **64%** | 100% |
| complex templates / peptide hits | **0 / 0** | **0 / 0** | 1 / 1 | **0 / 4** | 4 / 4 |

- **8RYP — the control for 8RYO.** Same peptide, same HLA-A\*03, same deposition date,
  different TCR. Both models land within 5–6° of the native and 0.69 Å of each other, where on
  8RYO AlphaFold 3 is 162° off. Nothing about the pMHC changed; the TCR did.
- **43RH — the hardest one both models solved.** Non-classical mouse H2-Q9, a TCR only 64%
  identical to anything pre-cutoff, no template complex, a peptide with no pre-cutoff hit, and
  a 3.33 Å native. Both still within 10° of the native.
- **8QFY — non-classical MHC, near-perfect.** HLA-E, the most accurate pair of the five.
  ESMFold2's ipTM is 0.80 here while it is essentially exact: the confidence understates it.
- **8WUL — a real neoantigen.** KRAS G12V VVGAVGVGK on HLA-A\*11, TCR at 64% identity, no
  template. Five sibling entries share this peptide and span 0.45–2.04 Å divergence, so it
  doubles as a series.
- **8PJG — the ceiling, and the only class II.** 0.945 both, 0.20 Å apart — but a 100%-identity
  TCR template exists, so it shows what "solved" looks like when the answer was available.

## Views

`make_views.py` writes `<PDB>/<PDB>.pml`; run `pymol 8RYO/8RYO.pml`. It loads all three
structures, colours them native gray / AF3 firebrick / ESMFold2 teal, and stores three scenes.
All four cases use the same construction, so the panels are directly comparable: **screen up =
MHC centre → native TCR centre, screen right = peptide N→C**. The native TCR is therefore
straight up in every panel and a prediction that docks elsewhere is off-axis by construction.

| scene | camera | use it for |
|---|---|---|
| `side` | across the groove | a TCR that moved to another surface — 8RYO |
| `top` | down the platform normal, from the TCR side | context |
| `footprint` | `top` with the TCR bodies hidden, one sphere per variable domain (big = Vα, small = Vβ) | a TCR that stayed on the groove and turned around — 9NW2, 9RU5, 9GV7 |

The `footprint` scene exists because an in-plane flip is invisible in `side` (the two TCRs
occupy the same envelope) and occluded in `top` (the TCR body covers the groove).
