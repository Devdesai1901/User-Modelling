import json
from collections import defaultdict
from statistics import mean

PATH = "hour9_steering_results.jsonl"

rows = [json.loads(l) for l in open(PATH, "r", encoding="utf-8")]

def avg(rows, key):
    vals = [r["metrics"][key] for r in rows]
    return mean(vals)

# group by (alpha, sign)
groups = defaultdict(list)
for r in rows:
    groups[(r["alpha"], r["sign"])].append(r)

# baseline group
base = groups[(0.0, "none")]

keys = ["n_words", "fk_grade", "fk_ease", "jargon_rate", "definition_count"]

print("Baseline averages:")
for k in keys:
    print(f"  {k}: {avg(base, k):.3f}")

print("\nSteering deltas vs baseline (avg steered - avg baseline):")
for (alpha, sign), g in sorted(groups.items()):
    if alpha == 0.0:
        continue
    print(f"\nalpha={alpha} sign={sign}")
    for k in keys:
        delta = avg(g, k) - avg(base, k)
        print(f"  Δ{k}: {delta:+.3f}")
