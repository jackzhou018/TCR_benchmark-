"""Figure 4: how the two generative samplers get to the answer, and what the schedule costs.

A  Ca RMSD to the native along the denoising trajectory, AF3 (200 steps) vs ESMFold2 (68),
   on the four complexes traced by layer_probe_trajectories/.
B  AF3 convergence to its own final frame, 30 complexes x 5 samples (overnight/).
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
a.set_title("A  Denoising trajectories (4 complexes)")
a.legend(frameon=False, loc="lower left")
a.axhline(10, ls=":", lw=1, c="#888")
a.text(.02, 11, "10 Å", fontsize=8, color="#666")

# B -- AF3 convergence to its own final frame
b = ax[1]
t = pd.read_csv(f"{BASE}/overnight/results/traj_rmsd_mean.csv")
b.plot(t.step, t.ca_rmsd_to_final_mean, color=BASE_COLORS["AF3"], lw=1.8, label="to final frame")
b.fill_between(t.step, t.ca_rmsd_to_final_median, t.ca_rmsd_to_final_p90,
               color=BASE_COLORS["AF3"], alpha=.2)
b.plot(t.step, t.ca_rmsd_to_native_mean, color="#555", lw=1.4, ls="--", label="to native")
b.set_yscale("log"); b.set_xlabel("AF3 denoising step (of 200)")
b.set_ylabel("mean Cα RMSD (Å)")
b.axvline(158, ls="--", lw=1.2, c="#b5545c")
b.text(152, 400, "step 158\n(< 1 Å)", fontsize=8, color="#b5545c", ha="right")
b.set_title("B  AF3 convergence to final frame (n = 30)")
b.legend(frameon=False, loc="lower left")

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
c.set_title("C  Accuracy vs GPU time by sampler setting")
c.annotate("", xy=(5.9, .692), xytext=(9.9, .692),
           arrowprops=dict(arrowstyle="->", color="#b5545c", lw=1.4))
c.text(7.9, .684, "1.80× faster,  ΔDockQ −0.010 (p = 0.33)",
       ha="center", fontsize=8.5, color="#b5545c")

fig.tight_layout()
fig.savefig(f"{OUT}/Figure_4.png", dpi=300)
print("wrote", f"{OUT}/Figure_4.png")
