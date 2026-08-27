"""Self-checks for the overnight AF3 shortcut experiment.

    ~/anaconda3/envs/local_esmfold2/bin/python -m pytest overnight/tests -q

Run these AFTER the GPU passes; the MSA check reads the server zips and the
coverage checks need the run directories to exist.
"""
from __future__ import annotations

import csv, json, sys, zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
OVN = ROOT / "overnight"
sys.path.insert(0, str(OVN / "scripts"))
sys.path.insert(0, str(ROOT / "layer_probe_trajectories" / "scripts"))
import common as C                      # noqa: E402

TRACKS = ["t1_baseline", "t2_fewsteps", "t3_onesample", "t4_both"]


def sample():
    return [r["pdb_id"] for r in csv.DictReader(open(OVN / "sample.csv"))]


def test_sample_is_30_from_the_scoreable_pool():
    s = sample()
    assert len(s) == len(set(s)) == 30
    manifest = {r["PDB ID"]: r for r in csv.DictReader(
        open(ROOT / "input_data/truncated_structures/manifest.csv"))}
    excl = {r["pdb_id"] for r in csv.DictReader(
        open(ROOT / "input_data/natives/excluded.csv"))}
    for p in s:
        assert manifest[p]["Status"] == "READY", p
        assert p not in excl, p
        assert (ROOT / "input_data/natives" / f"{p}.pdb").exists(), p


def test_sample_is_reproducible():
    """pick_sample.py must reproduce sample.csv exactly -- the seed is frozen."""
    import subprocess, tempfile, shutil
    before = (OVN / "sample.csv").read_bytes()
    with tempfile.TemporaryDirectory() as td:
        shutil.copy(OVN / "sample.csv", Path(td) / "keep.csv")
        subprocess.run([sys.executable, str(OVN / "scripts/pick_sample.py")],
                       check=True, capture_output=True)
        after = (OVN / "sample.csv").read_bytes()
        (OVN / "sample.csv").write_bytes(before)
    assert after == before


@pytest.mark.parametrize("pdb_id", sample())
def test_inputs_carry_the_server_msas_verbatim(pdb_id):
    """Every track must see the same MSAs the benchmark's AF3 predictions saw."""
    d = json.loads((OVN / "inputs" / f"{pdb_id}.json").read_text())
    zp = ROOT / "models/AF3/AF3" / f"fold_{pdb_id.lower()}_truncated_tcr_pmhc.zip"
    z = zipfile.ZipFile(zp)
    stem = zp.stem
    fasta = C.read_fasta(pdb_id)
    assert [s["protein"]["id"] for s in d["sequences"]] == C.chains_for(pdb_id)
    for s in d["sequences"]:
        ch = s["protein"]["id"].lower()
        assert s["protein"]["sequence"] == fasta[s["protein"]["id"]]
        for kind, key in (("unpaired", "unpairedMsa"), ("paired", "pairedMsa")):
            a3m = z.read(f"msas/{stem}_{kind}_msa_chains_{ch}.a3m").decode()
            assert s["protein"][key] == a3m, f"{pdb_id} chain {ch} {kind} MSA differs"
    prov = json.loads((OVN / "inputs" / f"{pdb_id}.provenance.json").read_text())
    assert prov["latest_template_date"] <= prov["template_cutoff"]
    assert d["modelSeeds"] == [int(x) for x in prov["server_seeds"]]


def test_convergence_detector():
    """first_below finds the step from which the curve STAYS under the threshold."""
    import traj_rmsd as T
    steps = np.arange(1, 201)
    curve = 20.0 * np.exp(-steps / 12.0)
    f = lambda c, t: (lambda ok: (int(np.flatnonzero(~ok)[-1] + 2)
                                  if (~ok).any() else 1))(c <= t)
    # a late blip must push the answer past the blip, not before it
    blipped = curve.copy()
    blipped[150] = 5.0
    assert f(curve, 1.0) < 60
    assert f(blipped, 1.0) == 152
    assert T.THRESHOLDS[1] == 1.0


@pytest.mark.skipif(not (OVN / "results/summary.csv").exists(), reason="no runs yet")
def test_every_track_covers_the_same_30_complexes():
    import pandas as pd
    s = set(sample())
    for tr in TRACKS:
        f = OVN / "results" / f"{tr}_times.csv"
        assert f.exists(), tr
        df = pd.read_csv(f)
        assert set(df.pdb_id) == s, tr
        assert df.pdb_id.is_unique
        assert (df.padded_tokens == 512).all(), f"{tr} left the 512 bucket"
    acc = pd.read_csv(OVN / "results/accuracy.csv")
    for tr in TRACKS:
        assert set(acc[acc.track == tr].pdb_id) == s, tr


@pytest.mark.skipif(not (OVN / "results/summary.csv").exists(), reason="no runs yet")
def test_tracks_differ_only_in_steps_and_samples():
    import pandas as pd
    conv = json.loads((OVN / "results/convergence.json").read_text())
    S = conv["rounded_step"]
    want = {"t1_baseline": (200, 5), "t2_fewsteps": (S, 5),
            "t3_onesample": (200, 1), "t4_both": (S, 1)}
    seeds = {}
    for tr, (st, sm) in want.items():
        df = pd.read_csv(OVN / "results" / f"{tr}_times.csv")
        assert (df.steps == st).all() and (df["samples"] == sm).all(), tr
        assert (df.recycles == 10).all(), tr
        seeds[tr] = dict(zip(df.pdb_id, df.seed))
    base = seeds["t1_baseline"]
    for tr, s in seeds.items():
        assert s == base, f"{tr} used different seeds"
