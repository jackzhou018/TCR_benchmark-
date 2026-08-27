#!/usr/bin/env python
"""Time and accuracy across the four tracks.

    ~/anaconda3/envs/local_esmfold2/bin/python figures.py

Inputs : results/<track>_times.csv (one row per prediction) and
         results/accuracy.csv (one row per track x complex).
Outputs: results/summary.csv, results/paired_deltas.csv,
         figures/02_time_by_track.png, figures/03_accuracy_by_track.png,
         figures/04_time_vs_accuracy.png

Everything is paired: the four tracks predict the SAME 30 complexes from the
SAME AlphaFold Server MSAs with the SAME seed, so per-complex differences are
the honest comparison and the paired deltas are what the middle panels show.
t0_trace is the instrumented pass; it is scored for parity but is never a
timing track.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "overnight" / "results"
FIG = ROOT / "overnight" / "figures"
ORDER = ["t1_baseline", "t2_fewsteps", "t3_onesample", "t4_both"]
COLOR = {"t1_baseline": "#4c72b0", "t2_fewsteps": "#dd8452",
         "t3_onesample": "#55a868", "t4_both": "#c44e52"}


def paired_ci(d: pd.Series, n_boot: int = 20000) -> dict:
    """Bootstrap 95% CI and a paired t-test on the per-complex difference.

    The across-complex SD is dominated by how hard each complex is, which is
    shared by every track; the paired difference is the only thing that
    isolates the sampler setting.
    """
    from scipy import stats

    x = d.values
    rng = np.random.default_rng(0)
    bs = rng.choice(x, (n_boot, len(x)), replace=True).mean(1)
    t = stats.ttest_rel(x, np.zeros_like(x))
    return dict(d_dockq_ci_lo=round(float(np.percentile(bs, 2.5)), 4),
                d_dockq_ci_hi=round(float(np.percentile(bs, 97.5)), 4),
                d_dockq_p_paired=round(float(t.pvalue), 4) if len(x) else float("nan"))


def load_times() -> pd.DataFrame:
    fs = sorted(RES.glob("*_times.csv"))
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    return df[df.track.isin(ORDER)].copy()


def label(df, track) -> str:
    r = df[df.track == track].iloc[0]
    return f"{int(r.steps)} steps\n{int(r['samples'])} sample{'s' if r['samples'] > 1 else ''}"


def main() -> None:
    t = load_times()
    a = pd.read_csv(RES / "accuracy.csv")
    tracks = [k for k in ORDER if k in set(t.track)]
    lab = {k: label(t, k) for k in tracks}
    FIG.mkdir(parents=True, exist_ok=True)

    base = "t1_baseline"
    piv_t = t.pivot(index="pdb_id", columns="track", values="inference_s")
    piv_tot = t.pivot(index="pdb_id", columns="track", values="total_s")
    piv_a = a[a.track.isin(tracks)].pivot(index="pdb_id", columns="track",
                                          values="global_dockq")

    # ---------------- summary table ----------------
    rows = []
    for k in tracks:
        r = t[t.track == k]
        ak = a[a.track == k]
        rows.append(dict(
            track=k, steps=int(r.steps.iloc[0]), samples=int(r["samples"].iloc[0]),
            n=len(r),
            inference_s_mean=round(r.inference_s.mean(), 2),
            inference_s_median=round(r.inference_s.median(), 2),
            inference_s_sd=round(r.inference_s.std(), 3),
            total_s_mean=round(r.total_s.mean(), 2),
            speedup_vs_baseline=round(piv_t[base].mean() / piv_t[k].mean(), 3),
            gpu_s_saved_per_prediction=round(piv_t[base].mean() - piv_t[k].mean(), 2),
            global_dockq_mean=round(ak.global_dockq.mean(), 4),
            global_dockq_median=round(ak.global_dockq.median(), 4),
            d_global_dockq_vs_baseline=round((piv_a[k] - piv_a[base]).mean(), 4),
            **paired_ci(piv_a[k] - piv_a[base]),
            ca_rmsd_mean=round(ak.ca_rmsd.mean(), 3),
            interface_rmsd_mean=round(ak.interface_rmsd.mean(), 3),
            tm_score_mean=round(ak.tm_score.mean(), 4),
            dockq_acceptable_frac=round((ak.global_dockq >= 0.23).mean(), 3),
            dockq_high_frac=round((ak.global_dockq >= 0.80).mean(), 3),
        ))
    summary = pd.DataFrame(rows)
    summary.to_csv(RES / "summary.csv", index=False)
    print(summary.to_string(index=False))

    deltas = pd.DataFrame({
        "pdb_id": piv_a.index,
        **{f"d_dockq_{k}": (piv_a[k] - piv_a[base]).values for k in tracks if k != base},
        **{f"speedup_{k}": (piv_t[base] / piv_t[k]).values for k in tracks if k != base},
    })
    deltas.to_csv(RES / "paired_deltas.csv", index=False)

    # ---------------- 02 time ----------------
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.3),
                           gridspec_kw={"width_ratios": [1.15, 1.15, 1]})
    data = [t[t.track == k].inference_s.values for k in tracks]
    bp = ax[0].boxplot(data, widths=0.55, showfliers=False, patch_artist=True,
                       medianprops=dict(color="k", lw=1.4))
    for p, k in zip(bp["boxes"], tracks):
        p.set_facecolor(COLOR[k]); p.set_alpha(0.75)
    for i, (d, k) in enumerate(zip(data, tracks), 1):
        ax[0].scatter(np.full(len(d), i) + np.random.default_rng(0).uniform(-.13, .13, len(d)),
                      d, s=9, color="k", alpha=0.45, zorder=3)
    ax[0].set_xticks(range(1, len(tracks) + 1))
    ax[0].set_xticklabels([lab[k] for k in tracks], fontsize=8)
    ax[0].set_ylabel("GPU inference time per prediction (s)")
    ax[0].set_title("Time per prediction")
    ax[0].set_ylim(bottom=0)

    # Where the time goes. All four tracks are one linear model in the diffusion
    # work done: t = trunk + c * (steps x samples). Fitting it over all 120
    # predictions is what shows why the speed-ups cap out below 2x.
    work = (t.steps * t["samples"]).values.astype(float)
    A = np.stack([np.ones_like(work), work], 1)
    trunk, c = np.linalg.lstsq(A, t.inference_s.values, rcond=None)[0]
    pred = A @ np.array([trunk, c])
    r2 = 1 - ((t.inference_s.values - pred) ** 2).sum() / \
        ((t.inference_s.values - t.inference_s.mean()) ** 2).sum()
    w = [int(t[t.track == k].steps.iloc[0] * t[t.track == k]["samples"].iloc[0])
         for k in tracks]
    ax[1].bar(range(len(tracks)), [trunk] * len(tracks), 0.6, color="#8c8c8c",
              alpha=0.9, label=f"trunk + overhead ({trunk:.2f} s, fixed)")
    ax[1].bar(range(len(tracks)), [c * x for x in w], 0.6, bottom=trunk,
              color=[COLOR[k] for k in tracks], alpha=0.9,
              label=f"diffusion ({c*1000:.2f} ms per sample-step)")
    for i, (k, x) in enumerate(zip(tracks, w)):
        ax[1].annotate(f"{piv_t[k].mean():.2f}s", (i, piv_t[k].mean()), ha="center",
                       va="bottom", fontsize=8.5)
    ax[1].axhline(trunk, color="k", ls=":", lw=1)
    ax[1].set_xticks(range(len(tracks)))
    ax[1].set_xticklabels([lab[k] for k in tracks], fontsize=8)
    ax[1].set_ylabel("mean inference time (s)")
    ax[1].set_title(f"Half the run is the trunk ($R^2$={r2:.3f})")
    ax[1].legend(frameon=False, fontsize=7.5, loc="upper right")
    ax[1].set_ylim(0, piv_t[base].mean() * 1.35)

    sp = [piv_t[base].mean() / piv_t[k].mean() for k in tracks]
    ax[2].bar(range(len(tracks)), sp, color=[COLOR[k] for k in tracks], alpha=0.85)
    for i, v in enumerate(sp):
        ax[2].annotate(f"{v:.2f}x", (i, v), ha="center", va="bottom", fontsize=9)
    ax[2].axhline(1, color="0.4", ls="--", lw=1)
    ax[2].set_xticks(range(len(tracks)))
    ax[2].set_xticklabels([lab[k] for k in tracks], fontsize=8)
    ax[2].set_ylabel("speed-up vs baseline")
    ax[2].set_title("Mean speed-up")
    for x in ax:
        x.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"AlphaFold 3 inference time, {len(piv_t)} TCR-pMHC complexes, "
                 "one RTX PRO 6000, serial", y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "02_time_by_track.png", dpi=200, bbox_inches="tight")

    # ---------------- 03 accuracy ----------------
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.3))
    ad = [a[a.track == k].global_dockq.values for k in tracks]
    bp = ax[0].boxplot(ad, widths=0.55, showfliers=False, patch_artist=True,
                       medianprops=dict(color="k", lw=1.4))
    for p, k in zip(bp["boxes"], tracks):
        p.set_facecolor(COLOR[k]); p.set_alpha(0.75)
    for i, (d, k) in enumerate(zip(ad, tracks), 1):
        ax[0].scatter(np.full(len(d), i) + np.random.default_rng(1).uniform(-.13, .13, len(d)),
                      d, s=9, color="k", alpha=0.45, zorder=3)
    ax[0].axhline(0.23, color="0.5", ls=":", lw=1)
    ax[0].annotate("DockQ 0.23 (acceptable)", (len(tracks) + 0.45, 0.235), ha="right",
                   fontsize=7.5, color="0.4")
    ax[0].set_xticks(range(1, len(tracks) + 1))
    ax[0].set_xticklabels([lab[k] for k in tracks], fontsize=8)
    ax[0].set_ylabel("Global DockQ")
    ax[0].set_title("Accuracy per track")

    others = [k for k in tracks if k != base]
    for i, k in enumerate(others):
        d = (piv_a[k] - piv_a[base]).values
        ax[1].scatter(np.full(len(d), i) + np.random.default_rng(2).uniform(-.13, .13, len(d)),
                      d, s=20, color=COLOR[k], alpha=0.8)
        ax[1].hlines(d.mean(), i - 0.3, i + 0.3, color="k", lw=2)
        ax[1].annotate(f"{d.mean():+.3f}", (i, d.mean()), textcoords="offset points",
                       xytext=(0, 8), ha="center", fontsize=9)
    ax[1].axhline(0, color="0.4", ls="--", lw=1)
    ax[1].set_xticks(range(len(others)))
    ax[1].set_xticklabels([lab[k] for k in others], fontsize=8)
    ax[1].set_ylabel(r"$\Delta$ Global DockQ vs baseline")
    ax[1].set_title("Paired change, same complex")

    for k in others:
        ax[2].scatter(piv_a[base], piv_a[k], s=22, color=COLOR[k], alpha=0.8,
                      label=lab[k].replace(chr(10), ", "))
    ax[2].plot([0, 1], [0, 1], color="0.4", ls="--", lw=1)
    ax[2].set_xlim(0, 1); ax[2].set_ylim(0, 1)
    ax[2].set_xlabel("baseline Global DockQ")
    ax[2].set_ylabel("track Global DockQ")
    ax[2].set_title("Paired, same complex")
    ax[2].legend(frameon=False, fontsize=7.5, loc="upper left")
    for x in ax:
        x.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Accuracy against the benchmark natives, {len(piv_a)} complexes "
                 "(full-atom DockQ v2)", y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "03_accuracy_by_track.png", dpi=200, bbox_inches="tight")

    # ---------------- 04 trade-off ----------------
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.3))
    for k in tracks:
        x, y = piv_t[k].mean(), piv_a[k].mean()
        d = piv_a[k] - piv_a[base]
        ci = paired_ci(d)
        lo, hi = y - (d.mean() - ci["d_dockq_ci_lo"]), y + (ci["d_dockq_ci_hi"] - d.mean())
        ax[0].errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt="o", ms=10,
                       color=COLOR[k], capsize=4, lw=1.6)
        off = {base: (-6, 30), "t2_fewsteps": (4, -36), "t3_onesample": (16, 4),
               "t4_both": (-10, -30)}.get(k, (0, 14))
        ax[0].annotate(f"{lab[k].replace(chr(10), ', ')}\n{y:.3f}",
                       (x, y), textcoords="offset points", xytext=off,
                       ha="right" if off[0] < 0 else "left", fontsize=8,
                       color=COLOR[k])
    ax[0].axhline(piv_a[base].mean(), color="0.6", ls=":", lw=1)
    ax[0].set_ylim(piv_a[base].mean() - 0.05, piv_a[base].mean() + 0.04)
    ax[0].set_xlabel("mean GPU inference time per prediction (s)")
    ax[0].set_ylabel("mean Global DockQ")
    ax[0].set_title("Cost vs accuracy (bars = 95% CI of the paired change)")
    ax[0].set_xlim(-0.5, 12.5)

    w = 0.38
    xs = np.arange(len(tracks))
    ax2 = ax[1].twinx()
    ax[1].bar(xs - w / 2, [piv_t[k].mean() for k in tracks], w, color="#4c72b0",
              alpha=0.85, label="time (s)")
    ax2.bar(xs + w / 2, [piv_a[k].mean() for k in tracks], w, color="#55a868",
            alpha=0.85, label="Global DockQ")
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels([lab[k] for k in tracks], fontsize=8)
    ax[1].set_ylabel("mean time (s)", color="#4c72b0")
    ax2.set_ylabel("mean Global DockQ", color="#55a868")
    ax2.set_ylim(0, 1)
    ax[1].set_title("Side by side")
    for x in (*ax, ax2):
        x.spines[["top"]].set_visible(False)
    ax[0].spines[["right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "04_time_vs_accuracy.png", dpi=200, bbox_inches="tight")

    conv = json.loads((RES / "convergence.json").read_text()) \
        if (RES / "convergence.json").exists() else {}
    print(json.dumps({"converged_step": conv.get("converged_step"),
                      "rounded_step": conv.get("rounded_step")}, indent=2))
    print("->", FIG)


if __name__ == "__main__":
    main()
