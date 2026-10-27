#!/usr/bin/env python3
"""Export the Sim(3)-aligned top-view trajectories of the reported benchmark runs.

Re-uses evaluate() of evaluate.py, so the alignment is exactly the one behind the reported metrics.
Run from configs/gs_agrifield_benchmark with the evaluation venv, after evaluate.py and aggregate.py:

    .venv_eval/bin/python paper/export_trajectories.py results
Writes results/trajectories_xy.csv (sequence, method, kind, x, y); kind = reference | estimate.
"""
import os, sys
import pandas as pd

out = sys.argv[1] if len(sys.argv) > 1 else "results"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import evaluate as E  # noqa: E402

pairs = pd.read_csv(os.path.join(out, "per_pair.csv"))
runs = E.find_runs(os.environ.get("TAG", "gsaf_benchmark")).set_index(["sequence", "method", "rep"])
pairs = pairs[pairs.apply(lambda p: (p.sequence, p.method, p.rep) in runs.index, axis=1)]
pairs["run_dir"] = [runs.loc[(p.sequence, p.method, p.rep), "path"] for _, p in pairs.iterrows()]
rows, done_ref = [], set()
for _, p in pairs.iterrows():
    r = E.evaluate(dict(sequence=p.sequence, method=p.method, rep=p.rep, path=p.run_dir))
    if "_est_xy" not in r:
        continue
    if p.sequence not in done_ref:
        rows += [(p.sequence, "reference", "reference", x, y) for x, y in r["_ref_xy"][::5]]
        done_ref.add(p.sequence)
    rows += [(p.sequence, p.method, "estimate", x, y) for x, y in r["_est_xy"][::5]]
if not rows:
    sys.exit("no run found for the pairs of per_pair.csv; nothing written")
pd.DataFrame(rows, columns=["sequence", "method", "kind", "x", "y"]).round(3).to_csv(
    os.path.join(out, "trajectories_xy.csv"), index=False)
print(len(rows), "points")
