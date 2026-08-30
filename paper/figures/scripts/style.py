"""Shared model palette for the paper figures.

Every panel imports COLORS from here, so the three models look the same in 1A/1B/2C/2D.
Dots are drawn in the lightened COLORS; trend lines and r-labels stay on the full-strength
BASE_COLORS, which keeps each fit readable on top of its own dots without changing hue.
"""
import glob
import matplotlib.colors as mcolors
import pandas as pd

MODELS = ["AF3", "Protenix", "ESMFold2"]
BASE_COLORS = {"AF3": "#4169E1", "Protenix": "#E8871A", "ESMFold2": "#2E9B45",
               "TCRmodel2": "#D62728"}
# TCRmodel2 is deliberately NOT in MODELS: 1A and 1B append it themselves. Its completed
# runnable Class I set contributes 111 of the 126 signed-off, scoreable structures
# (9J4S has no scoreable native interface); the colour lives here so 1A shares the palette.

# 9EJI, 9LLU and 9EJG/9EJH are deliberately NOT excluded: their TCR is docked, just not on the
# peptide, which is biology (9EJI is a published peptide-independent binder). 7SU9 is not
# excluded either -- its TCR was wrapped one -a unit cell away and build_natives.py re-images it.
# Natives with no scoreable interface at all (9J4S) never reach the results: run_dockq.py drops
# what input_data/natives/excluded.csv lists rather than scoring it.


def keep(df):
    """The signed-off rows -- review_status is set from manifest.csv's Status."""
    return df[df.review_status == "READY"]


LIGHTEN = 0.40   # fraction toward white -- the one knob for how pale the dots are


def lighten(c, f=LIGHTEN):
    return tuple(v + (1.0 - v) * f for v in mcolors.to_rgb(c))


COLORS = {m: lighten(c) for m, c in BASE_COLORS.items()}


BASE = "/14TBDrive/6TBDrive1_backup/benchmark_fresh"
KEYS = ["model", "config", "pdb_id"]


def load_dockq():
    """Every model+config's long per-interface CSV, concatenated."""
    return pd.concat([pd.read_csv(c) for c in
                      sorted(glob.glob(f"{BASE}/output/DockQ/*/*_dockq.csv"))], ignore_index=True)


def global_dockq(df, extra=()):
    """Per-structure Global DockQ -- DockQ v2's GlobalDockQ, its score over the whole complex.

    The one metric every panel plots. run_dockq.py writes it onto every interface row of a
    structure, so `.first()` per (model, config, pdb_id) is the per-structure value.
    """
    o = df.groupby(KEYS).global_dockq.first().rename("global_dockq")
    if extra:
        o = pd.concat([o, df.groupby(KEYS)[list(extra)].first()], axis=1)
    return o.reset_index().dropna(subset=["global_dockq"])
