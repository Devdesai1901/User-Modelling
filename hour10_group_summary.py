# hour10_group_summary.py
import json
from collections import defaultdict
from statistics import mean

PATH = "hour9_steering_results_scored.jsonl"

rows = [json.loads(l) for l in open(PATH, "r", encoding="utf-8")]
groups = defaultdict(list)
for r in rows:
    groups[(r["alpha"], r["sign"])].append(r["metrics_v2"])

keys = rows[0]["metrics_v2"].keys()

print("=== Group means (metrics_v2) ===")
for (alpha, sign), mets in sorted(groups.items()):
    print(f"\nalpha={alpha} sign={sign}")
    for k in keys:
        print(f"  {k}: {mean(m[k] for m in mets):.3f}")
