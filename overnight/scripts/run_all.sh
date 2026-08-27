#!/usr/bin/env bash
# Sequential GPU passes. One at a time, on GPU 0, nothing else running:
# the whole point is that the wall times are comparable.
#
#   bash run_all.sh early     # trajectory pass + the two tracks that don't need S*
#   bash run_all.sh late 50   # the two reduced-step tracks, once S* is known
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=(env -u LD_LIBRARY_PATH /home/yash/anaconda3/envs/af3/bin/python)   # clean LD_LIBRARY_PATH or JAX drops to CPU
LOG=../logs

run () {  # run <track> <steps> <samples> [--trace]
  echo "=== $1 steps=$2 samples=$3 ${4:-} @ $(date -Is)"
  "${PY[@]}" run_track.py --track "$1" --steps "$2" --samples "$3" ${4:-} \
      > "$LOG/$1.log" 2> "$LOG/$1.err"
  tail -3 "$LOG/$1.log"
}

case "${1:?usage: run_all.sh early|late [steps]}" in
  early)
    run t0_trace     200 5 --trace     # trajectories only; NOT a timing track
    run t1_baseline  200 5             # stock AF3 defaults
    run t3_onesample 200 1
    ;;
  late)
    S="${2:?late needs the rounded converged step count}"
    run "t2_fewsteps" "$S" 5
    run "t4_both"     "$S" 1
    ;;
esac
echo "done $(date -Is)"
