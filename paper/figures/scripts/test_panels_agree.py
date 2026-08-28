"""Every panel must plot the same Global DockQ, for the same structures."""
import pandas as pd
from style import keep, load_dockq, global_dockq

d = global_dockq(keep(load_dockq()), ["resolution", "mhc_class"])
ref = d.set_index(["model", "config", "pdb_id"]).global_dockq

assert not ref.isna().any()
assert ref.between(0, 1).all()

# 126 signed-off structures: 136 entries - 9 still in review - 9J4S (no scoreable interface,
# dropped by run_dockq.py, never scored). A model may be short if it lacks predictions, but it
# must never carry a structure outside that set.
n = d.groupby("model").pdb_id.nunique()
full = set(d.pdb_id)
assert len(full) == 126, len(full)
assert n.max() == 126, n
behind = n[n < 126]

# 2C is the only panel that drops anything further: entries with no deposited resolution.
res = pd.to_numeric(d.resolution, errors="coerce")
assert res.notna().all(), f"2C would drop {(~res.notna()).sum()} entries the other panels keep"

# every panel must plot global_dockq, not the TCR-only interface mean.
g = keep(load_dockq()).groupby(["model", "config", "pdb_id"]).global_dockq.first()
assert ref.equals(g.rename("global_dockq"))

print(f"ok  {len(full)} structures, range {ref.min():.3f}-{ref.max():.3f}"
      + (f"; behind: {behind.to_dict()}" if len(behind) else "; all models complete"))
