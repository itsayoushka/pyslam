#!/usr/bin/env bash
# GS-AgriField baseline benchmark — retry-until-success protocol.
# For every (sequence, method): attempt 1 (seed 12345); only if it fails (not completed or < 90 % frames tracked),
# attempt 2 (seed 23456), then attempt 3 (seed 34567). The first successful attempt is the reported run; the number
# of attempts is reported too. Attempt-major order: all attempt-1 runs first, so an interrupted weekend still has
# one attempt for every pair. Safe to re-run (finished attempts: logs/*.DONE, successes: logs/*.SUCCESS).
#   usage: run_all.sh [filter]        e.g. run_all.sh __sift      STOP_LOST (default 50) consecutive lost frames
set -u
B=$(cd "$(dirname "$0")" && pwd)
LOGS=$B/logs
PY=${EVAL_PYTHON:-$B/.venv_eval/bin/python}   # Python with evo 1.31.1, numpy, pandas, scipy
mkdir -p "$LOGS"
export STOP_LOST=${STOP_LOST:-50}
FILTER=${1:-}
[[ -f $B/versions/environment.txt ]] || "$B/capture_versions.sh"

for rep in 1 2 3; do
  for method in orb2 sift_lightglue akaze superpoint_lightglue; do
    for cfg in "$B"/runs/rep${rep}__*__${method}.json; do
      [[ -e "$cfg" ]] || continue
      name=$(basename "$cfg" .json)               # rep<k>__<id>_<seqname>__<method>
      pair=${name#rep${rep}__}                     # <id>_<seqname>__<method>
      seq=${pair%__*}
      [[ -n "$FILTER" && "$name" != *"$FILTER"* ]] && continue
      [[ -f "$LOGS/$pair.SUCCESS" ]] && continue   # an earlier attempt succeeded
      [[ -f "$LOGS/$name.DONE" ]] && continue
      echo "$(date '+%F %T')  start $name"
      start=$(date +%s)
      "$B/run_docker.sh" "$cfg" > "$LOGS/$name.log" 2>&1
      rc=$?
      echo "rc=$rc start=$start end=$(date +%s)" > "$LOGS/$name.DONE"
      if "$PY" "$B/evaluate.py" --check "$seq" "$method" "$rep" >> "$LOGS/$name.log" 2>&1; then
        echo "attempt=$rep" > "$LOGS/$pair.SUCCESS"; res=SUCCESS
      else
        res=FAILURE
      fi
      echo "$(date '+%F %T')  end   $name rc=$rc $(( ($(date +%s) - start) / 60 )) min  $res"
    done
  done
done
"$PY" "$B/evaluate.py" --plot && "$PY" "$B/aggregate.py"
