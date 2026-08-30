# Four cases for the divergence figure

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
