"""Figure 4: how the two generative samplers get to the answer, and what the schedule costs.

A  Ca RMSD to the native along the denoising trajectory, AF3 (200 steps) vs ESMFold2 (68),
   on the four complexes traced by layer_probe_trajectories/.
B  Global DockQ along the same trajectories: when each sampler's interface becomes correct.
C  Wall-clock vs accuracy for the four sampler tracks (overnight/results/summary.csv).
"""
import glob, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from style import BASE_COLORS

BASE = "/14TBDrive/6TBDrive1_backup/benchmark_fresh"
OUT = f"{BASE}/paper/figures/outputs"
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9})
COL = {"af3": BASE_COLORS["AF3"], "esmfold2": BASE_COLORS["ESMFold2"]}
NAME = {"af3": "AlphaFold 3", "esmfold2": "ESMFold2"}

fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.5))

# A -- trajectories
a = ax[0]
seen = set()
for f in sorted(glob.glob(f"{BASE}/layer_probe_trajectories/outputs/*/*/*_diffusion_frames.csv")):
    d = pd.read_csv(f).sort_values("frame")
    m = d.model.iloc[0]
    a.plot(d.normalized_progress, d.ca_rmsd, color=COL[m], lw=1.4, alpha=.8,
           label=NAME[m] if m not in seen else None)
    seen.add(m)
a.set_yscale("log"); a.set_xlabel("fraction of denoising steps")
a.set_ylabel("Cα RMSD to native (Å)")
a.set_title(r"$\bf{A}$  Denoising trajectories (4 complexes)", loc="left")
a.legend(frameon=False, loc="lower left")
a.axhline(10, ls=":", lw=1, c="#888")
a.text(.02, 11, "10 Å", fontsize=8, color="#666")

# B -- when the interface becomes correct, both samplers
b = ax[1]
GRID = np.linspace(0, 1, 201)
frames = {}
for f in sorted(glob.glob(f"{BASE}/layer_probe_trajectories/outputs/*/*/*_diffusion_frames.csv")):
    d = pd.read_csv(f)
    if "global_dockq" not in d or d.global_dockq.isna().all():
        continue                      # 7SG2 has no per-frame DockQ
    frames.setdefault(d.model.iloc[0], []).append(
        d.dropna(subset=["global_dockq"]).sort_values("normalized_progress"))
for m, runs in frames.items():
    curves = np.vstack([np.interp(GRID, d.normalized_progress, d.global_dockq) for d in runs])
    for c in curves:                                   # individual complexes, as texture
        b.plot(GRID, c, color=COL[m], lw=.8, alpha=.3, zorder=2)
    med = np.median(curves, axis=0)
    b.plot(GRID, med, color=COL[m], lw=2.4, zorder=4, label=f"{NAME[m]} (n = {len(runs)})")
    fin = med[-1]
    x = GRID[np.argmax(med >= .9 * fin)]               # median crossing of 90% of final
    b.plot([x], [.9 * fin], "o", color=COL[m], ms=7, mec="white", mew=1.2, zorder=6)
    b.annotate(f"{x*100:.0f}%", (x, .9 * fin), textcoords="offset points",
               xytext=(0, 11), ha="center", fontsize=9, color=COL[m], fontweight="bold")
b.axhline(0.49, ls="--", lw=1, c="#b5545c")
b.text(.02, .51, "medium", fontsize=8, color="#b5545c")
b.set_xlabel("fraction of denoising steps"); b.set_ylabel("Global DockQ")
b.set_ylim(-0.02, 1.0); b.set_xlim(0, 1.02)
b.set_title(r"$\bf{B}$  When the interface becomes correct", loc="left")
b.legend(frameon=False, loc="upper left")
b.text(.98, .03, "markers: 90% of the run's own final DockQ",
       transform=b.transAxes, ha="right", fontsize=7.5, color="#666")

# C -- time vs accuracy for the four tracks
c = ax[2]
s = pd.read_csv(f"{BASE}/overnight/results/summary.csv")
lab = {"t1_baseline": "200×5\n(default)", "t2_fewsteps": "160×5",
       "t3_onesample": "200×1", "t4_both": "160×1"}
for _, r in s.iterrows():
    c.errorbar(r.inference_s_mean, r.global_dockq_mean, fmt="o", ms=9,
               color=BASE_COLORS["AF3"], alpha=.85)
    c.annotate(lab[r.track], (r.inference_s_mean, r.global_dockq_mean),
               textcoords="offset points", xytext=(0, 11), ha="center", fontsize=8.5)
base = s[s.track == "t1_baseline"].iloc[0]
c.axhline(base.global_dockq_mean, ls=":", lw=1, c="#888")
c.set_xlim(4.4, 11.4); c.set_ylim(0.66, 0.775)
c.set_xlabel("GPU s per prediction"); c.set_ylabel("mean Global DockQ")
c.set_title(r"$\bf{C}$  Accuracy vs GPU time by sampler", loc="left")
c.annotate("", xy=(5.9, .692), xytext=(9.9, .692),
           arrowprops=dict(arrowstyle="->", color="#b5545c", lw=1.4))
c.text(7.9, .684, "1.80× faster,  ΔDockQ −0.010 (p = 0.33)",
       ha="center", fontsize=8.5, color="#b5545c")

fig.tight_layout()
fig.savefig(f"{OUT}/Figure_sampler.png", dpi=300)
print("wrote", f"{OUT}/Figure_sampler.png")
