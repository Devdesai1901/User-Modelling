# hour17_controls_score.py
import json
from statistics import mean
from collections import defaultdict
from scipy.stats import wilcoxon

from metrics import explanation_metrics

IN_PATH = "hour17_controls.jsonl"

FIELDS = ["fk_grade", "fk_ease", "definition_count", "jargon_rate", "n_words"]

def paired_report(name, deltas):
    deltas = [d for d in deltas if d is not None]
    if len(deltas) < 5:
        return f"{name}: insufficient"
    stat, p = wilcoxon(deltas)
    return f"{name:28s} mean={mean(deltas):+.3f} median={sorted(deltas)[len(deltas)//2]:+.3f} p={p:.3g}"

def main():
    rows = [json.loads(l) for l in open(IN_PATH, "r", encoding="utf-8") if l.strip()]
    print(f"Loaded {len(rows)} rows\n")

    # Collect deltas
    D = defaultdict(list)

    for r in rows:
        # compute metrics for each text
        m = {}
        for k in ["novice_base","novice_plusv","novice_minusv","novice_rand","novice_zero",
                  "expert_base","expert_minusv","expert_rand","expert_zero"]:
            m[k] = explanation_metrics(r[k])

        # deltas vs baseline
        for met in FIELDS:
            D[f"novice_plusv.{met}"].append(m["novice_plusv"][met] - m["novice_base"][met])
            D[f"novice_minusv.{met}"].append(m["novice_minusv"][met] - m["novice_base"][met])
            D[f"novice_rand.{met}"].append(m["novice_rand"][met] - m["novice_base"][met])
            D[f"novice_zero.{met}"].append(m["novice_zero"][met] - m["novice_base"][met])

            D[f"expert_minusv.{met}"].append(m["expert_minusv"][met] - m["expert_base"][met])
            D[f"expert_rand.{met}"].append(m["expert_rand"][met] - m["expert_base"][met])
            D[f"expert_zero.{met}"].append(m["expert_zero"][met] - m["expert_base"][met])

    # Print summaries
    print("=== NOVICE controls (deltas vs novice_base) ===")
    for met in FIELDS:
        print("\nMetric:", met)
        print(paired_report("novice_zero (should ~0)", D[f"novice_zero.{met}"]))
        print(paired_report("novice_rand (should ~0)", D[f"novice_rand.{met}"]))
        print(paired_report("novice_minusv (flip)", D[f"novice_minusv.{met}"]))
        print(paired_report("novice_plusv  (+v)", D[f"novice_plusv.{met}"]))

    print("\n=== EXPERT controls (deltas vs expert_base) ===")
    for met in FIELDS:
        print("\nMetric:", met)
        print(paired_report("expert_zero (should ~0)", D[f"expert_zero.{met}"]))
        print(paired_report("expert_rand (should ~0)", D[f"expert_rand.{met}"]))
        print(paired_report("expert_minusv (-v)", D[f"expert_minusv.{met}"]))

if __name__ == "__main__":
    main()
