"""Merge single-seed replay outputs (one per training order) into one results file.

usage: python scripts/merge_seeds.py out.json part0.json part1.json ...
"""
import json, sys

out_path, parts = sys.argv[1], sys.argv[2:]
docs = [json.load(open(p, encoding="utf-8")) for p in parts]
docs.sort(key=lambda d: d.get("seed", 0) or 0)
base = docs[0]
merged_repo = None
for d in docs:
    r = d["repos"][0]
    if merged_repo is None:
        merged_repo = {k: v for k, v in r.items() if k not in ("curves", "learn_stats", "graph_summary")}
        merged_repo.update({"curves": {}, "learn_stats": [], "graph_summary": []})
    for arm, curves in r["curves"].items():
        merged_repo["curves"].setdefault(arm, []).extend(curves)
    merged_repo["learn_stats"].extend(r.get("learn_stats", []))
    merged_repo["graph_summary"].extend(r.get("graph_summary", []))
out = {k: v for k, v in base.items() if k not in ("repos", "seed")}
out["seeds"] = len(docs)
out["repos"] = [merged_repo]
json.dump(out, open(out_path, "w", encoding="utf-8"), indent=1)
print(f"{out_path}: {len(docs)} seeds, store={out['store']}, arms={sorted(merged_repo['curves'])}")
