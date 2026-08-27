"""Build alphafold3-dialect input JSON from the benchmark's AF3 Server bundles.

The benchmark ran AF3 on AlphaFold Server, which cannot expose activations, but
its result zips carry the *inputs* the server used: the paired and unpaired
MSAs and the template hits with their query->template index maps. Reusing them
means the local AF3 run sees exactly the benchmark's data-pipeline output --
closer to "match the benchmark inference protocol" than re-running a local
search against different database snapshots would be, and it inherits the
server's 2021-09-30 template cutoff (verified in tests/test_af3_inputs.py).
"""
from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

import common as C

ZIP_DIR = C.REPO / "models" / "AF3" / "AF3"
CUTOFF = "2021-09-30"


def zip_path(pdb_id: str) -> Path:
    return ZIP_DIR / f"fold_{pdb_id.lower()}_truncated_tcr_pmhc.zip"


def template_dates(cif_text: str) -> list[str]:
    return sorted(set(re.findall(r"\b(?:19|20)\d\d-\d\d-\d\d\b", cif_text)))


def build(pdb_id: str, *, seed: int | None = None,
          drop_templates=False) -> tuple[dict, dict]:
    zp = zip_path(pdb_id)
    if not zp.exists():
        raise FileNotFoundError(zp)
    z = zipfile.ZipFile(zp)
    stem = zp.stem                                   # fold_7rtr_truncated_tcr_pmhc
    req = json.loads(z.read(f"{stem}_job_request.json"))[0]
    fasta = C.read_fasta(pdb_id)
    chains = C.chains_for(pdb_id)
    if len(req["sequences"]) != len(chains):
        raise ValueError(f"{pdb_id}: {len(req['sequences'])} server chains "
                         f"!= {len(chains)} benchmark chains")

    seqs = []
    latest = "0000-00-00"
    for ch, entry in zip(chains, req["sequences"]):
        server_seq = entry["proteinChain"]["sequence"]
        if server_seq != fasta[ch]:
            raise ValueError(
                f"{pdb_id} chain {ch}: server input != benchmark FASTA"
            )
        low = ch.lower()
        unpaired = z.read(f"msas/{stem}_unpaired_msa_chains_{low}.a3m").decode()
        paired = z.read(f"msas/{stem}_paired_msa_chains_{low}.a3m").decode()

        templates = []
        hits_name = f"templates/{stem}_template_hits_chains_{low}_query_to_hit.json"
        if not drop_templates and hits_name in z.namelist():
            for hit in json.loads(z.read(hits_name)):
                cif = z.read(f"templates/{hit['name']}").decode()
                dates = template_dates(cif)
                if dates:
                    latest = max(latest, dates[-1])
                templates.append({
                    "mmcif": cif,
                    "queryIndices": hit["queryIndices"],
                    "templateIndices": hit["templateIndices"],
                })
        seqs.append({"protein": {
            "id": ch, "sequence": server_seq,
            "unpairedMsa": unpaired, "pairedMsa": paired,
            "templates": templates,
        }})

    if latest > CUTOFF:
        raise ValueError(
            f"{pdb_id}: template dated {latest} is past the {CUTOFF} cutoff"
        )

    seeds = [int(s) for s in req["modelSeeds"]] if seed is None else [int(seed)]
    return {
        "name": pdb_id,
        "modelSeeds": seeds,
        "sequences": seqs,
        "dialect": "alphafold3",
        "version": 3,
    }, {
        # AF3 rejects unknown keys in the input JSON, so provenance goes to a
        # sidecar rather than into the document itself.
        "source_zip": str(zp),
        "source_zip_sha256": C.sha256(zp),
        "server_seeds": req["modelSeeds"],
        "latest_template_date": latest,
        "template_cutoff": CUTOFF,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=C.CACHE / "af3_inputs")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ok, bad = 0, []
    for p in args.pdb:
        try:
            d, prov = build(p, seed=args.seed)
            (args.out / f"{p}.json").write_text(json.dumps(d))
            (args.out / f"{p}.provenance.json").write_text(
                json.dumps(prov, indent=2))
            ok += 1
        except Exception as e:                                  # noqa: BLE001
            bad.append((p, str(e)[:120]))
    print(f"wrote {ok} inputs to {args.out}")
    for p, e in bad:
        print(f"  SKIP {p}: {e}")


if __name__ == "__main__":
    main()
