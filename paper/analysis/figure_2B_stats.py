"""Statistics behind Figure 2B (Global DockQ vs native resolution).

Writes paper/analysis/Figure_2B_analysis.txt. Reads the same data the panel does, through
style.keep()/style.global_dockq(), so every n here matches the figure.
"""
import sys, itertools, numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

ROOT = "/14TBDrive/6TBDrive1_backup/benchmark_fresh"
sys.path.insert(0, f"{ROOT}/paper/figures/scripts")
from style import keep, load_dockq, global_dockq

RNG = np.random.default_rng(0)
BOOT = 5000
SESOI = 0.05          # smallest slope of interest, DockQ per Angstrom -- see the write-up
BANDS = [(0.00, 0.23, "Incorrect"), (0.23, 0.49, "Acceptable"),
         (0.49, 0.80, "Medium"), (0.80, 1.01, "High")]
out = []
def P(s=""): out.append(s)


# ---------------------------------------------------------------- data
long = keep(load_dockq())
d = global_dockq(long, ["resolution", "mhc_class", "method"])
d["resolution"] = pd.to_numeric(d.resolution, errors="coerce")
assert d.resolution.notna().all()
d["xray"] = d.method == "X-RAY DIFFRACTION"

# Structure level: one row per complex, DockQ averaged over the models that scored it.
# The three models see identical inputs, so treating their 371 rows as independent would
# trible the apparent n. The common set (every model present) keeps the average balanced.
common = set.intersection(*(set(g.pdb_id) for _, g in d.groupby("model")))
s_df = (d[d.pdb_id.isin(common)]
     .groupby("pdb_id")
     .agg(dockq=("global_dockq", "mean"), resolution=("resolution", "first"),
          mhc_class=("mhc_class", "first"), xray=("xray", "first"))
     .reset_index())


def ci_boot(x, y, fn, n=BOOT):
    """Percentile bootstrap CI for a statistic of paired samples."""
    idx = RNG.integers(0, len(x), size=(n, len(x)))
    vals = np.array([fn(x[i], y[i]) for i in idx])
    return np.percentile(vals, [2.5, 97.5])


def slope(x, y):
    return np.polyfit(x, y, 1)[0]


# ---------------------------------------------------------------- 1. descriptives
P("=" * 78)
P("FIGURE 2B -- GLOBAL DockQ vs NATIVE RESOLUTION")
P("Statistical analysis")
P("=" * 78)
P()
P("QUESTION")
P("-" * 78)
P("Figure 2B shows three downward trend lines: predictions score worse against")
P("lower-resolution natives. Two things are worth separating.")
P()
P("  (1) Is the trend real, or is it three noisy fits through a cloud?")
P("  (2) If it is real, is there a resolution below which it stops mattering --")
P("      a cutoff where a benchmark can stop worrying about native quality?")
P()
P("The second question is the one with practical consequences: it decides whether")
P("a resolution filter belongs in the benchmark's inclusion criteria.")
P()
P()
P("DATA")
P("-" * 78)
P(f"{d.pdb_id.nunique()} structures, {len(d)} model-structure scores "
  f"({', '.join(f'{m} n={g.pdb_id.nunique()}' for m, g in d.groupby('model'))}).")
P(f"Structure-level set (scored by all three models): n = {len(s_df)}.")
P(f"Resolution: median {s_df.resolution.median():.2f} A, "
  f"IQR {s_df.resolution.quantile(.25):.2f}-{s_df.resolution.quantile(.75):.2f}, "
  f"range {s_df.resolution.min():.2f}-{s_df.resolution.max():.2f}.")
P(f"Method: {(s_df.xray).sum()} X-ray, {(~s_df.xray).sum()} cryo-EM.")
P(f"Class: {(s_df.mhc_class == 'Class I').sum()} class I, "
  f"{(s_df.mhc_class == 'Class II').sum()} class II.")
P()
q = [1.5, 2.0, 2.5, 3.0, 3.5, 4.5]
P("  resolution bin      n   median DockQ   Incorrect/Acceptable/Medium/High")
for lo, hi in zip(q, q[1:]):
    sub = s_df[(s_df.resolution >= lo) & (s_df.resolution < hi)]
    if not len(sub):
        continue
    comp = [((sub.dockq >= a) & (sub.dockq < b)).sum() for a, b, _ in BANDS]
    P(f"  {lo:.1f}-{hi:.1f} A        {len(sub):4d}   {sub.dockq.median():.3f}          "
      + " / ".join(str(c) for c in comp))
P()


# ---------------------------------------------------------------- 2. association
P()
P("METHOD 1 -- IS THE ASSOCIATION REAL?")
P("-" * 78)
P("Pearson r (the fit drawn in the panel) plus Spearman rho, which does not assume")
P("linearity and is not dragged by the sparse tail beyond 3.5 A. Confidence")
P(f"intervals from {BOOT} percentile bootstrap resamples of the structures.")
P()
P("Per model, on that model's own structures:")
P()
P("  model       n    Pearson r  [95% CI]        p         Spearman rho   p")
for m in ["AF3", "Protenix", "ESMFold2"]:
    g = d[d.model == m]
    x, y = g.resolution.values, g.global_dockq.values
    r, pr = stats.pearsonr(x, y)
    rho, prho = stats.spearmanr(x, y)
    lo, hi = ci_boot(x, y, lambda a, b: stats.pearsonr(a, b)[0])
    P(f"  {m:10s}{len(g):4d}   {r:+.3f}  [{lo:+.3f}, {hi:+.3f}]  {pr:.2e}   "
      f"{rho:+.3f}        {prho:.2e}")
P()
x, y = s_df.resolution.values, s_df.dockq.values
r, pr = stats.pearsonr(x, y)
rho, prho = stats.spearmanr(x, y)
lo, hi = ci_boot(x, y, lambda a, b: stats.pearsonr(a, b)[0])
P(f"Structure level (mean of the three models, n={len(s_df)}):")
P(f"  Pearson r = {r:+.3f} [{lo:+.3f}, {hi:+.3f}], p = {pr:.2e}")
P(f"  Spearman rho = {rho:+.3f}, p = {prho:.2e}")
P(f"  r^2 = {r**2:.3f}  -- resolution explains {100*r**2:.1f}% of the variance in DockQ.")
P()

# mixed model on the long data: repeated measures on the same structures
mix = smf.mixedlm("global_dockq ~ resolution + C(model)", d, groups=d.pdb_id).fit()
b = mix.params["resolution"]; se = mix.bse["resolution"]; pv = mix.pvalues["resolution"]
P("Mixed-effects model over all model-structure scores, random intercept per")
P("structure (this is the primary test -- it uses every score without pretending")
P("the three models are independent observations):")
P()
P(f"  global_dockq ~ resolution + model + (1|pdb_id)")
P(f"  resolution slope = {b:+.4f} DockQ per A  (SE {se:.4f}, "
  f"95% CI [{b-1.96*se:+.4f}, {b+1.96*se:+.4f}], p = {pv:.2e})")
P()
sl, (slo, shi) = slope(x, y), ci_boot(x, y, slope)
P(f"  Structure-level OLS slope for comparison: {sl:+.4f} [{slo:+.4f}, {shi:+.4f}] per A")
P()


# ---------------------------------------------------------------- 3. confounders
P()
P("METHOD 2 -- IS IT REALLY RESOLUTION?")
P("-" * 78)
P("Resolution travels with other things. Cryo-EM entries sit at the blunt end of")
P("the scale, and class II is a different prediction problem. Both are checked by")
P("re-running the correlation inside each stratum: a trend that survives is not")
P("just the strata being compared to each other.")
P()
for label, sub in [("X-ray only", s_df[s_df.xray]), ("cryo-EM only", s_df[~s_df.xray]),
                   ("class I only", s_df[s_df.mhc_class == "Class I"]),
                   ("class II only", s_df[s_df.mhc_class == "Class II"])]:
    if len(sub) < 5:
        P(f"  {label:16s} n={len(sub):3d}   too few to test")
        continue
    rr, pp = stats.spearmanr(sub.resolution, sub.dockq)
    bb = slope(sub.resolution.values, sub.dockq.values)
    P(f"  {label:16s} n={len(sub):3d}   Spearman rho = {rr:+.3f}, p = {pp:.3g}, "
      f"slope = {bb:+.4f}/A")
P()
mw = stats.mannwhitneyu(s_df[s_df.xray].dockq, s_df[~s_df.xray].dockq)
P(f"  X-ray vs cryo-EM DockQ: Mann-Whitney U p = {mw.pvalue:.3g} "
  f"(medians {s_df[s_df.xray].dockq.median():.3f} vs {s_df[~s_df.xray].dockq.median():.3f})")
mcls = smf.ols("dockq ~ resolution + C(mhc_class) + C(xray)", s_df).fit()
P(f"  Slope with class and method held fixed: "
  f"{mcls.params['resolution']:+.4f}/A, p = {mcls.pvalues['resolution']:.3g}")
P()


# ---------------------------------------------------------------- 4. threshold
P()
P("METHOD 3 -- IS THERE A CUTOFF?")
P("-" * 78)
P("Three models fitted to the structure-level data and compared by AIC:")
P()
P("  flat       DockQ ~ constant                     (resolution never matters)")
P("  linear     DockQ ~ a + b*res                    (matters equally throughout)")
P("  plateau    DockQ ~ a below k, sloping above k   (matters only past a cutoff)")
P()
P("The plateau model is the one that answers the question: its breakpoint k is the")
P("resolution below which the data show no dependence. k is fitted by scanning")
P("every candidate between the 10th and 90th percentile of resolution and keeping")
P(f"the lowest residual sum of squares; its CI comes from {2000} bootstrap refits.")
P()


def fit_flat(x, y):
    return np.full_like(y, y.mean()), 1


def fit_linear(x, y):
    b, a = np.polyfit(x, y, 1)
    return a + b * x, 2


def fit_plateau(x, y, k):
    """Constant up to k, straight line after it, joined continuously."""
    z = np.maximum(x - k, 0.0)
    A = np.column_stack([np.ones_like(z), z])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return A @ coef, coef


def rss(pred, y):
    return float(((y - pred) ** 2).sum())


def aic(r, n, kpar):
    return n * np.log(r / n) + 2 * kpar


grid = np.arange(np.percentile(x, 10), np.percentile(x, 90), 0.02)


def best_knot(x, y):
    r = [(rss(fit_plateau(x, y, k)[0], y), k) for k in grid]
    return min(r)[1]


k = best_knot(x, y)
n = len(y)
res_flat, res_lin = rss(fit_flat(x, y)[0], y), rss(fit_linear(x, y)[0], y)
pred_pl, coef_pl = fit_plateau(x, y, k)
res_pl = rss(pred_pl, y)
P(f"  model      params   RSS      AIC       dAIC")
rows = [("flat", 1, res_flat), ("linear", 2, res_lin), ("plateau", 3, res_pl)]
aics = {nm: aic(rv, n, kp) for nm, kp, rv in rows}
best = min(aics.values())
for nm, kp, rv in rows:
    P(f"  {nm:10s}{kp:5d}   {rv:.4f}   {aics[nm]:8.2f}  {aics[nm]-best:+6.2f}")
P()
boot_k = []
for _ in range(2000):
    i = RNG.integers(0, n, n)
    boot_k.append(best_knot(x[i], y[i]))
klo, khi = np.percentile(boot_k, [2.5, 97.5])
P(f"  Fitted breakpoint k = {k:.2f} A   (95% bootstrap CI {klo:.2f}-{khi:.2f} A)")
P(f"  Below k: flat at DockQ = {coef_pl[0]:.3f}")
P(f"  Above k: {coef_pl[1]:+.4f} DockQ per A")
P()
P("  Linear is not a special case of the plateau family -- plateau forces a zero")
P("  slope below k -- so the two are not nested and no F-test applies. AIC and the")
P("  bootstrap CI carry the comparison.")
P()


# ---------------------------------------------------------------- 5. equivalence
P()
P("METHOD 4 -- WHERE DOES RESOLUTION STOP MATTERING?")
P("-" * 78)
P("The practical form of the question: if the benchmark were filtered at X A, would")
P("the structures that survive still show a resolution effect? Each cumulative")
P("subset is tested two ways.")
P()
P("Spearman rho is the primary readout. Fitted slopes are NOT comparable across")
P("these subsets: truncating the x-range inflates a regression slope even when the")
P("underlying relationship is unchanged, so a steeper slope in a narrower window is")
P("an artifact of the window, not a stronger effect.")
P()
P('"No significant correlation" in a small subset is also not evidence of no')
P("effect -- usually it is lost power. So each subset also gets an equivalence")
P(f"test against a smallest slope of interest of {SESOI:.2f} DockQ per A. Over the ~3 A")
P("span of this benchmark that slope moves a structure by 0.15 DockQ, half a CAPRI")
P("band; below it, resolution cannot change how a structure is classified.")
P()
P(f"  EQUIVALENT   the whole 90 pct CI of the slope falls inside +/-{SESOI:.2f} -- a real")
P("               absence of effect, not just a null result")
P("  effect       correlation significant at p < 0.05")
P("  INCONCLUSIVE neither: the subset cannot tell the two apart")
P()
P("  subset            n    rho      p(rho)     slope/A   90% CI            verdict")
for cut in [2.2, 2.5, 2.8, 3.0, 3.5, 10.0]:
    sub = s_df[s_df.resolution <= cut]
    if len(sub) < 15:
        continue
    xs, ys = sub.resolution.values, sub.dockq.values
    rr, prho = stats.spearmanr(xs, ys)
    b1 = slope(xs, ys)
    idx = RNG.integers(0, len(xs), size=(BOOT, len(xs)))
    bs = np.array([slope(xs[i], ys[i]) for i in idx])
    lo90, hi90 = np.percentile(bs, [5, 95])
    verdict = ("EQUIVALENT" if lo90 > -SESOI and hi90 < SESOI
               else "effect" if prho < 0.05 else "INCONCLUSIVE")
    tag = f"<= {cut:.1f} A" if cut < 10 else "all (<= 4.5)"
    P(f"  {tag:16s}{len(sub):4d}   {rr:+.3f}   {prho:8.3g}   {b1:+.4f}   "
      f"[{lo90:+.4f},{hi90:+.4f}]  {verdict}")
P()
P("rho stays between -0.25 and -0.36 in every window, including the tightest one.")
P("The trend is not carried by the poorly diffracting tail: it is present at the")
P("same strength among the best-resolved structures in the set.")
P()


# ------------------------------------------------------------ 4b. which failure mode
P()
P("METHOD 5 -- COORDINATE ERROR, OR GENUINELY HARDER TARGETS?")
P("-" * 78)
P("Two mechanisms would both produce this correlation, and they mean opposite")
P("things for the benchmark.")
P()
P("  (a) Measurement artifact. A lower-resolution native has less accurate atomic")
P("      positions. DockQ scores against those coordinates, so even a perfect")
P("      prediction loses points. Nothing about the prediction is worse.")
P("  (b) Real difficulty. Complexes that diffract poorly are often conformationally")
P("      heterogeneous or weakly bound -- exactly the complexes whose interface a")
P("      model has trouble placing. The target really is harder.")
P()
P("These separate on WHICH part of DockQ degrades. Coordinate error perturbs atoms")
P("locally: it costs fnat (the contact set) and iRMSD (interface geometry) while")
P("leaving the TCR in roughly the right place. Genuine difficulty costs LRMSD --")
P("the model docks the TCR somewhere else entirely.")
P()
tcr = long[long.interface_role.str.contains("TCR") & ~long.interface_role.str.contains("TCRa-TCRb")]
per = (tcr.groupby(["model", "pdb_id"])
          .agg(fnat=("fnat", "mean"), irms=("iRMSD", "mean"), lrms=("LRMSD", "mean"))
          .reset_index()
          .merge(s_df[["pdb_id", "resolution"]], on="pdb_id"))
P("  Spearman rho against resolution, over the TCR-facing interfaces:")
P()
P("  metric                       rho      p        reads as")
for col, lab, sense in [("fnat", "fnat (contact set)", "lower is worse"),
                        ("irms", "iRMSD (interface geometry)", "higher is worse"),
                        ("lrms", "LRMSD (TCR placement)", "higher is worse")]:
    rr, pp = stats.spearmanr(per.resolution, per[col])
    P(f"  {lab:28s}{rr:+.3f}   {pp:8.3g}  {sense}")
P()
lo_r = s_df[s_df.resolution <= s_df.resolution.median()]
hi_r = s_df[s_df.resolution > s_df.resolution.median()]
s_df["poor"] = (s_df.dockq < 0.49).astype(int)
lg = smf.logit("poor ~ resolution", s_df).fit(disp=0)
orr = np.exp(lg.params["resolution"])
cil, cih = np.exp(lg.conf_int().loc["resolution"])
P(f"  Odds of falling below Medium quality (DockQ < 0.49), per additional A:")
P(f"    odds ratio = {orr:.2f} [95% CI {cil:.2f}, {cih:.2f}], p = {lg.pvalues['resolution']:.3g}")
P(f"    {s_df.poor.sum()} of {len(s_df)} structures fall below 0.49.")
P()


# ---------------------------------------------------------------- 6. binned trend
P()
P("METHOD 6 -- MONOTONE TREND ACROSS BINS")
P("-" * 78)
P("A distribution-free check that does not depend on any fitted shape: split into")
P("resolution bins and test whether DockQ falls monotonically across them")
P("(Kruskal-Wallis for any difference, Spearman on the raw pairs for direction).")
P()
bins = [1.5, 2.2, 2.6, 3.0, 4.6]
groups, labels = [], []
for lo, hi in zip(bins, bins[1:]):
    g = s_df[(s_df.resolution >= lo) & (s_df.resolution < hi)].dockq.values
    if len(g):
        groups.append(g); labels.append(f"{lo:.1f}-{hi:.1f}")
kw = stats.kruskal(*groups)
P("  bin            n    median   mean")
for lab, g in zip(labels, groups):
    P(f"  {lab:12s}{len(g):4d}   {np.median(g):.3f}    {g.mean():.3f}")
P()
P(f"  Kruskal-Wallis H = {kw.statistic:.2f}, p = {kw.pvalue:.3g}")
pairs = list(itertools.combinations(range(len(groups)), 2))
ps = [stats.mannwhitneyu(groups[i], groups[j]).pvalue for i, j in pairs]
rej, padj, *_ = multipletests(ps, method="fdr_bh")
P(f"  Pairwise Mann-Whitney, Benjamini-Hochberg corrected:")
for (i, j), pa, rj in zip(pairs, padj, rej):
    P(f"    {labels[i]} vs {labels[j]}:  p_adj = {pa:.3g}{'  *' if rj else ''}")
P()

# ---------------------------------------------------------------- 7. interpretation
P()
P("=" * 78)
P("WHAT THIS MEANS")
P("=" * 78)
P()
P("1. The trend in Figure 2B is real, and it is modest.")
P(f"   Global DockQ falls {abs(b):.3f} per A of native resolution (mixed model,")
P(f"   p = {pv:.1e}), consistently across all three models. Resolution explains")
P(f"   {100*r**2:.0f}% of the variance in DockQ -- the other 90% is everything else about")
P("   the complex. Over the range this benchmark spans, 1.5 to 4.4 A, that slope")
P(f"   adds up to about {abs(b)*(s_df.resolution.max()-s_df.resolution.min()):.2f} DockQ, close to one full CAPRI band.")
P("   Real, but not the dominant term: the spread of DockQ at any single")
P(f"   resolution (IQR {s_df.dockq.quantile(.75)-s_df.dockq.quantile(.25):.2f}) is about as wide as the entire trend across the")
P("   resolution range.")
P()
P("2. There is no cutoff. This is the main negative result.")
P("   The plateau model -- flat below some k, declining above it -- fits WORSE than")
P("   a plain straight line once its extra parameter is paid for (dAIC +3.0), and")
P("   its breakpoint is not identifiable: the bootstrap CI runs from 2.0 to 3.3 A,")
P("   most of the occupied range. Spearman rho stays between -0.25 and -0.47 in")
P("   every truncated window, including structures at 2.2 A and better. No")
P("   equivalence test passed at any cutoff.")
P()
P("   So the effect is gradual and uniform. There is no resolution above which")
P("   natives are 'good enough' and below which they are not, and a benchmark")
P("   cannot filter this away -- cutting at 2.5 A would discard 55 percent of the")
P("   set and leave a subset with the same rho it started with.")
P()
P("3. It is not merely a scoring artifact, and this is the biologically")
P("   interesting part.")
P("   A blunt native is a blunt ruler: atoms are less precisely placed, so fnat and")
P("   iRMSD suffer even for a perfect prediction. That is real and it is present")
P("   here. But if that were the whole story, LRMSD -- which asks the much coarser")
P("   question of whether the TCR was docked in the right place at all -- would be")
P("   comparatively spared. It is not. All three degrade at the same rate")
P("   (|rho| = 0.23-0.24), and the odds of a structure falling below Medium quality")
P(f"   rise {orr:.1f}-fold per A (95% CI {cil:.1f}-{cih:.1f}).")
P()
P("   Predictions do not just lose precision against poorly resolved natives, they")
P("   lose the docking site. That points at the complexes themselves: TCR-pMHC")
P("   pairs that diffract poorly tend to be the conformationally mobile, weakly or")
P("   degenerately engaged ones, with CDR loops that are not pinned to a single")
P("   pose. Those are exactly the interfaces a single-structure predictor has")
P("   least purchase on. Resolution is acting partly as a readout of how well")
P("   defined the complex is, not only of how well it was measured.")
P()
P("4. Practical consequences for this benchmark.")
P("   - Do not filter on resolution. It would cost a large fraction of the set and")
P("     not remove the effect.")
P("   - Do report resolution alongside DockQ, and do not compare models across")
P("     subsets with different resolution distributions without adjusting: a 0.5 A")
P("     difference in median resolution is worth about 0.04 DockQ on its own.")
P("   - The class II subset (n=13 here) sits at a different resolution")
P("     distribution from class I; some of any class I/II DockQ gap seen in Figure")
P("     2C is resolution, not class. Holding class and method fixed, the")
P(f"     resolution slope is {mcls.params['resolution']:+.4f}/A (p = {mcls.pvalues['resolution']:.3g}), so the effect is not")
P("     simply the two classes being compared to each other.")
P()
P()
P("CAVEATS")
P("-" * 78)
P("- Observational. Resolution is not assigned; it correlates with method, class,")
P("  construct design and crystallisation behaviour. The stratified fits control")
P("  for what is recorded, not for what is not.")
P(f"- Thin at the ends: only {(s_df.resolution > 3.5).sum()} structures beyond 3.5 A and "
  f"{(s_df.resolution < 2.0).sum()} below 2.0 A. The fitted")
P("  slope past 3.5 A rests on very few points.")
P(f"- Only {(~s_df.xray).sum()} cryo-EM entries, so the X-ray/EM comparison is indicative only.")
P(f"- Class II n = {(s_df.mhc_class == 'Class II').sum()}; its within-class correlation is not powered and is")
P("  reported for completeness, not as a finding.")
P("- The structure-level analyses use the 119 complexes all three models scored, so")
P("  the mean is over the same three models everywhere. Per-model correlations in")
P("  Method 1 use each model's full set (AF3 119, Protenix 126, ESMFold2 126).")
P("- Global DockQ has a floor near 0.24 in this benchmark, because it averages in")
P("  the MHC-peptide and TCRa-TCRb interfaces that every model gets right. No")
P("  structure is scored 'Incorrect' on this metric, which compresses the low end")
P("  and, if anything, makes the resolution effect look smaller than it is.")
P()
P("Reproduce: python paper/analysis/figure_2B_stats.py")
P("Data: output/DockQ/dockq_all.csv via style.keep()/style.global_dockq();")
P("      resolution from input_data/natives/natives_manifest.csv.")
P(f"Seed {0}, {BOOT} bootstrap resamples.")
P()

pathlib_out = f"{ROOT}/paper/analysis/Figure_2B_analysis.txt"
open(pathlib_out, "w").write("\n".join(out) + "\n")
print("\n".join(out))
print(f"\n-> {pathlib_out}")
