"""Deposition and release dates for the 126 targets, straight from the RCSB entry API."""
import json, urllib.request, concurrent.futures as cf
from pathlib import Path
import pandas as pd
ROOT = Path("/14TBDrive/6TBDrive1_backup/benchmark_fresh")
t = pd.read_csv(ROOT / "analysis/training_sequence_similarity/sequence_similarity_per_target.csv")
def get(pid):
    d = json.load(urllib.request.urlopen(f"https://data.rcsb.org/rest/v1/core/entry/{pid}", timeout=60))
    a = d["rcsb_accession_info"]
    return {"pdb_id": pid, "deposit_date": a["deposit_date"][:10],
            "release_date_rcsb": a["initial_release_date"][:10],
            "revision_date": a.get("revision_date", "")[:10]}
with cf.ThreadPoolExecutor(8) as ex:
    rows = list(ex.map(get, t.pdb_id))
D = pd.DataFrame(rows)
D = D.merge(t[["pdb_id", "release_date"]], on="pdb_id")
D["date_matches_repo"] = D.release_date_rcsb == D.release_date
D.to_csv(Path(__file__).parent / "dates.csv", index=False)
print(D.date_matches_repo.value_counts().to_dict())
print("released on or before 2021-09-30:", int((D.release_date_rcsb <= "2021-09-30").sum()))
print("deposited on or before 2021-09-30:", int((D.deposit_date <= "2021-09-30").sum()))
print("earliest release:", D.release_date_rcsb.min(), " latest:", D.release_date_rcsb.max())
