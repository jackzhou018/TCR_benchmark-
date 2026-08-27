"""Shared config, native handling, distance bins and pair sampling.

Chain convention is the benchmark's (see ../CLAUDE.md), used verbatim:
class I -> A = MHC-I a1a2, B = peptide, C = TCR alpha V, D = TCR beta V.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
WORK = REPO / "layer_probe_trajectories"
INPUT_DATA = REPO / "input_data"
FASTA_DIR = INPUT_DATA / "truncated_structures" / "fastas"
MANIFEST = INPUT_DATA / "truncated_structures" / "manifest.csv"
CHAIN_MANIFEST = INPUT_DATA / "truncated_structures" / "chain_manifest.csv"
NATIVE_DIR = INPUT_DATA / "natives"
CACHE = WORK / "cache"
OUT = WORK / "outputs"

TARGET = "7RTR"
CLASS_I_CHAINS = ("A", "B", "C", "D")
# Class II inserts MHC-II beta1 as C, which pushes the TCR to D/E.
CLASS_II_CHAINS = ("A", "B", "C", "D", "E")
# Receptor / ligand grouping for the interface metrics and DockQ.
PMHC = ("A", "B")
TCR = ("C", "D")
PMHC_II = ("A", "B", "C")
TCR_II = ("D", "E")


def chains_for(pdb_id: str) -> list[str]:
    """Benchmark chain letters present in the FASTA, in convention order."""
    fasta = read_fasta(pdb_id)
    chains = [ch for ch in CLASS_II_CHAINS if ch in fasta]
    extra = set(fasta) - set(chains)
    if extra:
        raise ValueError(f"{pdb_id}: unexpected chain(s) {sorted(extra)}")
    return chains


def groups_for(labels) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(pMHC, TCR) grouping implied by the chain letters actually present.

    Class II is the 5-chain case, so chain E is the discriminator.
    """
    return (PMHC_II, TCR_II) if "E" in set(labels) else (PMHC, TCR)

# ---------------------------------------------------------------------------
# Distance bins -- IDENTICAL for AF3 and ESMFold2. 34 bins:
#   bin 0        : d <  4.0 A
#   bins 1..32   : 0.5 A wide, 4.0 -> 20.0 A
#   bin 33       : d >= 20.0 A
# ---------------------------------------------------------------------------
BIN_EDGES = np.arange(4.0, 20.0 + 1e-9, 0.5)          # 33 edges
NUM_BINS = len(BIN_EDGES) + 1                          # 34
# Representative centre of each bin, used for the expected-distance readout.
BIN_CENTERS = np.concatenate(
    [[3.5], (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2.0, [20.5], [21.5]]
)[:NUM_BINS]

# Pair-sampling categories. Balanced sampling stops the |i-j|<=8 backbone
# pairs -- which are ~free to predict -- from swamping the interface signal.
PAIR_CATEGORIES = (
    "intra_short",      # same chain, |i-j| <= 8
    "intra_medium",     # same chain, 8 < |i-j| <= 24
    "intra_long",       # same chain, |i-j| > 24
    "inter_pmhc",       # A <-> B
    "inter_tcr",        # C <-> D
    "inter_pmhc_tcr",   # (A|B) <-> (C|D)   <- the interface we care about
)


def digitize(d: np.ndarray) -> np.ndarray:
    """Ca-Ca distance (A) -> bin index in [0, NUM_BINS)."""
    return np.digitize(d, BIN_EDGES).astype(np.int64)


def stable_seed(s: str) -> int:
    """Process-independent seed from a string (builtin hash() is randomised)."""
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def sha256(path: Path, limit: int | None = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
            if limit and h.digest_size and fh.tell() > limit:
                break
    return h.hexdigest()


# ---------------------------------------------------------------------------
# FASTA / native
# ---------------------------------------------------------------------------
def read_fasta(pdb_id: str) -> dict[str, str]:
    """{'A': seq, 'B': seq, ...} keyed by benchmark chain letter."""
    path = FASTA_DIR / f"{pdb_id}_truncated.fasta"
    seqs, name = {}, None
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            name = line[1:].split("_")[-1]          # 'Chain_A' -> 'A'
            seqs[name] = ""
        else:
            seqs[name] += line
    return seqs


THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O",
}


def read_native_ca(pdb_id: str) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Ca coords + one-letter sequence per benchmark chain from the built native.

    The benchmark's natives are already cropped to the FASTA and already carry
    benchmark chain letters, so this is a plain read -- no remapping.
    """
    path = NATIVE_DIR / f"{pdb_id}.pdb"
    ca: dict[str, list] = {}
    seq: dict[str, list] = {}
    seen: dict[str, set] = {}
    for line in path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[12:16].strip() != "CA":
            continue
        alt = line[16]
        if alt not in (" ", "A"):
            continue
        ch = line[21]
        resid = line[22:27]                      # resseq + icode
        if resid in seen.setdefault(ch, set()):
            continue
        seen[ch].add(resid)
        ca.setdefault(ch, []).append(
            (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        )
        seq.setdefault(ch, []).append(THREE_TO_ONE.get(line[17:20].strip(), "X"))
    return (
        {k: np.asarray(v, dtype=np.float64) for k, v in ca.items()},
        {k: "".join(v) for k, v in seq.items()},
    )


def _map_native_to_fasta(native_seq: str, fasta_seq: str) -> list[int]:
    """FASTA index for each native residue, or -1 if it aligns to a gap.

    Natives are cropped to the FASTA but residues without density are simply
    absent, and the numbering does not have to start at the FASTA's first
    residue (7PBC's chain A starts at the FASTA's second residue). Aligning is
    the only safe way to place them; an off-by-one here would mislabel every
    probe target for that complex.
    """
    if native_seq == fasta_seq:
        return list(range(len(fasta_seq)))
    from Bio import Align

    al = Align.PairwiseAligner(mode="global", match_score=2, mismatch_score=-1,
                               open_gap_score=-10, extend_gap_score=-0.5,
                               end_gap_score=0.0)
    aln = al.align(fasta_seq, native_seq)[0]
    out = [-1] * len(native_seq)
    for (fs, fe), (ns, ne) in zip(aln.aligned[0], aln.aligned[1]):
        for k in range(fe - fs):
            out[ns + k] = fs + k
    return out


def native_ca_stack(pdb_id: str, chains=CLASS_I_CHAINS):
    """Ca coords laid out on the FASTA, plus which positions are resolved.

    Returns (xyz [N,3] with NaN at unresolved positions, chain labels [N],
    present mask [N]). N is always the FASTA length, so probe targets and
    model activations share one indexing.
    """
    ca, seq = read_native_ca(pdb_id)
    fasta = read_fasta(pdb_id)
    xyz, labels, present = [], [], []
    for ch in chains:
        n = len(fasta[ch])
        if ch not in ca:
            raise ValueError(f"{pdb_id}: native has no chain {ch}")
        idx = _map_native_to_fasta(seq[ch], fasta[ch])
        block = np.full((n, 3), np.nan)
        mask = np.zeros(n, dtype=bool)
        for k, fi in enumerate(idx):
            if fi < 0:
                continue
            if seq[ch][k] != fasta[ch][fi]:
                raise ValueError(
                    f"{pdb_id} chain {ch}: residue {k} aligns to a mismatch"
                )
            block[fi] = ca[ch][k]
            mask[fi] = True
        if mask.sum() < 0.8 * n:
            raise ValueError(
                f"{pdb_id} chain {ch}: only {int(mask.sum())}/{n} residues resolved"
            )
        xyz.append(block)
        labels += [ch] * n
        present.append(mask)
    return (np.concatenate(xyz, 0), np.array(labels),
            np.concatenate(present))


def chain_slices(labels: np.ndarray) -> dict[str, slice]:
    out = {}
    for ch in dict.fromkeys(labels.tolist()):
        idx = np.flatnonzero(labels == ch)
        out[ch] = slice(int(idx[0]), int(idx[-1]) + 1)
    return out


# ---------------------------------------------------------------------------
# Balanced pair sampling
# ---------------------------------------------------------------------------
def categorize_pairs(labels: np.ndarray) -> dict[str, np.ndarray]:
    """Category -> (M,2) array of i<j residue-index pairs."""
    n = len(labels)
    iu, ju = np.triu_indices(n, k=1)
    same = labels[iu] == labels[ju]
    sep = np.abs(iu - ju)
    in_p = np.isin(labels[iu], PMHC)
    in_t = np.isin(labels[iu], TCR)
    jn_p = np.isin(labels[ju], PMHC)
    jn_t = np.isin(labels[ju], TCR)

    masks = {
        "intra_short": same & (sep <= 8),
        "intra_medium": same & (sep > 8) & (sep <= 24),
        "intra_long": same & (sep > 24),
        "inter_pmhc": ~same & in_p & jn_p,
        "inter_tcr": ~same & in_t & jn_t,
        "inter_pmhc_tcr": ~same & ((in_p & jn_t) | (in_t & jn_p)),
    }
    return {k: np.stack([iu[m], ju[m]], 1) for k, m in masks.items()}


NEAR_CUT = 15.0   # A; the near/far stratum boundary used when sampling


def sample_pairs(labels: np.ndarray, per_category: int, seed: int,
                 present: np.ndarray | None = None,
                 native_xyz: np.ndarray | None = None,
                 near_cut: float = NEAR_CUT) -> np.ndarray:
    """(K,2) reproducible, doubly balanced sample of residue pairs.

    Balanced twice over:

    * across the six contact categories, so short-range backbone pairs cannot
      swamp the interface, and
    * within each category, half from d < `near_cut` and half from d >=
      `near_cut`.

    The second axis matters more than it looks: 93.7% of 7RTR's (A|B)x(C|D)
    pairs are further apart than 20 A and only 0.07% are contacts, so an
    unstratified draw of 500 interface pairs contains essentially no contacts
    and a probe fitted on it just learns "far". Sampling on the label like
    this shifts the class prior, which `bin_prior()` measures and the
    reconstruction undoes with a constant added to the logits -- an affine
    correction, so the probe stays strictly linear.

    `present` drops pairs touching an unresolved native residue.
    """
    rng = np.random.default_rng(seed)
    cats = categorize_pairs(labels)
    if present is not None:
        cats = {k: (v[present[v[:, 0]] & present[v[:, 1]]] if len(v) else v)
                for k, v in cats.items()}
    picked = []
    for cat in PAIR_CATEGORIES:
        pool = cats[cat]
        if len(pool) == 0:
            continue
        if native_xyz is None:
            strata = [pool]
        else:
            d = np.linalg.norm(
                native_xyz[pool[:, 0]] - native_xyz[pool[:, 1]], axis=-1)
            strata = [pool[d < near_cut], pool[d >= near_cut]]
        want = [per_category // 2, per_category - per_category // 2][:len(strata)]
        if len(strata) == 1:
            want = [per_category]
        take = [min(w, len(st)) for w, st in zip(want, strata)]
        # A short stratum tops up from the other so the category total holds.
        deficit = per_category - sum(take)
        for i in range(len(strata)):
            if deficit <= 0:
                break
            extra = min(deficit, len(strata[i]) - take[i])
            take[i] += extra
            deficit -= extra
        for st, k in zip(strata, take):
            if k:
                picked.append(st[rng.choice(len(st), size=k, replace=False)])
    return np.concatenate(picked, 0)


def bin_prior(labels: np.ndarray, native_xyz: np.ndarray,
              present: np.ndarray | None = None) -> np.ndarray:
    """Bin histogram over ALL valid residue pairs of one complex.

    This is the distribution the reconstruction actually faces (it applies the
    probe to every pair), as opposed to the stratified distribution the probe
    was fitted on.
    """
    n = len(labels)
    iu, ju = np.triu_indices(n, k=1)
    if present is not None:
        keep = present[iu] & present[ju]
        iu, ju = iu[keep], ju[keep]
    d = np.linalg.norm(native_xyz[iu] - native_xyz[ju], axis=-1)
    return np.bincount(digitize(d), minlength=NUM_BINS).astype(np.float64)


def pair_category_ids(labels: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Category index per sampled pair, for per-category probe metrics."""
    cats = categorize_pairs(labels)
    lookup = {}
    for ci, cat in enumerate(PAIR_CATEGORIES):
        for i, j in cats[cat]:
            lookup[(int(i), int(j))] = ci
    return np.array([lookup[(int(i), int(j))] for i, j in pairs], dtype=np.int64)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))


def read_cif_ca(path: Path, chains=CLASS_I_CHAINS):
    """Ca coords per chain from an mmCIF written by either model.

    Uses label_asym_id when auth_asym_id is absent; both models emit the
    benchmark letters, so no remapping is done or allowed here.
    """
    text = Path(path).read_text().splitlines()
    cols, rows, in_loop, header = {}, [], False, []
    for line in text:
        s = line.strip()
        if s.startswith("_atom_site."):
            header.append(s.split(".", 1)[1])
            in_loop = True
            continue
        if in_loop:
            if s.startswith("#") or s.startswith("loop_") or not s:
                break
            rows.append(s.split())
    cols = {name: i for i, name in enumerate(header)}
    ch_key = "auth_asym_id" if "auth_asym_id" in cols else "label_asym_id"
    seq_key = "auth_seq_id" if "auth_seq_id" in cols else "label_seq_id"
    ca: dict[str, list] = {}
    seen: dict[str, set] = {}
    for r in rows:
        if len(r) < len(header):
            continue
        if r[cols["label_atom_id"]].strip('"') != "CA":
            continue
        if r[cols["group_PDB"]] not in ("ATOM", "HETATM"):
            continue
        ch = r[cols[ch_key]]
        key = r[cols[seq_key]]
        if key in seen.setdefault(ch, set()):
            continue
        seen[ch].add(key)
        ca.setdefault(ch, []).append(
            (float(r[cols["Cartn_x"]]), float(r[cols["Cartn_y"]]),
             float(r[cols["Cartn_z"]]))
        )
    missing = [c for c in chains if c not in ca]
    if missing:
        raise ValueError(f"{path}: missing chains {missing}; got {sorted(ca)}")
    return np.concatenate([np.asarray(ca[c], dtype=np.float64) for c in chains])
