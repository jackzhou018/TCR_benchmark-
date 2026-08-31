"""Download the 259 pre-cutoff reference entries (mmCIF.gz) named by the reference set."""
import gzip, urllib.request, concurrent.futures as cf
from pathlib import Path
import pandas as pd
ROOT = Path("/14TBDrive/6TBDrive1_backup/benchmark_fresh")
OUT = Path(__file__).parent / "ref_cif"
ids = sorted(pd.read_csv(ROOT / "analysis/training_sequence_similarity/training_reference_set.csv").pdb_id.unique())

def get(pid):
    p = OUT / f"{pid}.cif"
    if p.exists() and p.stat().st_size > 1000:
        return pid, "cached"
    try:
        with urllib.request.urlopen(f"https://files.rcsb.org/download/{pid}.cif.gz", timeout=60) as r:
            p.write_bytes(gzip.decompress(r.read()))
        return pid, "ok"
    except Exception as e:
        return pid, f"FAIL {e}"

with cf.ThreadPoolExecutor(8) as ex:
    res = list(ex.map(get, ids))
bad = [r for r in res if r[1].startswith("FAIL")]
print(f"{len(ids)} entries, {sum(1 for _,s in res if s=='ok')} downloaded, "
      f"{sum(1 for _,s in res if s=='cached')} cached, {len(bad)} failed")
for b in bad[:10]: print(" ", b)
