#!/usr/bin/env bash
# Diffusion (sampling) trajectory for one complex -> outputs/<PDB>/<model>/
#
#     bash scripts/run_diffusion.sh 8RYO            # both models
#     bash scripts/run_diffusion.sh 8RYO af3        # one model
#
# No probes and no activation pool are involved: this is the model's own
# sampler, patched to record the coordinate state after every denoising step.
# Re-uses cache/<model>_target/<PDB> if the inference already ran.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PDB="${1:?usage: run_diffusion.sh <PDB_ID> [esmfold2|af3]}"
# 9J4S has no scoreable native interface (input_data/natives/excluded.csv), so it is
# out of the benchmark's 126 and every per-frame metric here would be meaningless.
grep -q "^$PDB," ../../input_data/natives/excluded.csv 2>/dev/null && {
  echo "$PDB is in input_data/natives/excluded.csv -- no scoreable interface, refusing"; exit 1; }
MODELS="${2:-esmfold2 af3}"
ESM="${ESM:-/home/yash/anaconda3/envs/local_esmfold2/bin/python}"
AF3="${AF3:-/home/yash/anaconda3/envs/af3/bin/python}"
# AF3's JAX needs a clean LD_LIBRARY_PATH or it silently falls back to CPU.
AF3RUN=(env -u LD_LIBRARY_PATH "$AF3")

for M in $MODELS; do
  if [ ! -f "../cache/${M}_target/$PDB/diffusion.npz" ]; then
    echo "== $M inference: $PDB"
    if [ "$M" = esmfold2 ]; then
      "$ESM" esmfold2_runner.py --pdb "$PDB" --mode full --num-loops 10 \
          --num-steps 100 --trace-diffusion --out ../cache/esmfold2_target
    else
      [ -f "../cache/af3_inputs/$PDB.json" ] || \
          "$ESM" af3_inputs.py --pdb "$PDB" --out ../cache/af3_inputs
      # pad to the complex's own length, not the 512 pool bucket
      N=$("$ESM" -c "import sys;sys.path.insert(0,'.');import common as C;\
print(sum(len(s) for s in C.read_fasta('$PDB').values()))")
      "${AF3RUN[@]}" af3_runner.py --pdb "$PDB" --mode full --num-recycles 10 \
          --num-samples 5 \
          --bucket "$N" --out ../cache/af3_target
    fi
  fi
  "$ESM" diffusion_traj.py --model "$M" --pdb "$PDB" \
      --cache "../cache/${M}_target/$PDB" --out "../outputs/$PDB/$M"
done
echo "-> outputs/$PDB/{$(echo $MODELS | tr ' ' ,)}/*_diffusion.xtc"
