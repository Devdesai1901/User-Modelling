# hour14_postpatch_metrics.py
import json
import csv
from statistics import mean
from collections import defaultdict

from metrics import explanation_metrics

IN_PATH = "hour13_steered.jsonl"
OUT_JSONL = "hour14_scored.jsonl"
OUT_CSV = "hour14_compare.csv"

FIELDS = [
    "n_words", "fk_grade", "fk_ease", "jargon_rate",
    "definition_count", "example_count", "bullet_count", "math_symbol_rate"
]

def main():
    rows = [json.loads(l) for l in open(IN_PATH, "r", encoding="utf-8") if l.strip()]
    print(f"Loaded {len(rows)} rows from {IN_PATH}")

    csv_rows = []
    agg = defaultdict(list)

    with open(OUT_JSONL, "w", encoding="utf-8") as out_f:
        for r in rows:
            m_nb = explanation_metrics(r["novice_base"])
            m_np = explanation_metrics(r["novice_plusv"])
            m_eb = explanation_metrics(r["expert_base"])
            m_em = explanation_metrics(r["expert_minusv"])

            # Save enriched JSONL
            r_out = dict(r)
            r_out["metrics_v2"] = {
                "novice_base": m_nb,
                "novice_plusv": m_np,
                "expert_base": m_eb,
                "expert_minusv": m_em,
            }
            out_f.write(json.dumps(r_out, ensure_ascii=False) + "\n")

            # Build CSV row (flat + deltas)
            row = {
                "idx": r["idx"],
                "topic": r.get("topic", ""),
                "domain": r.get("domain", ""),
                "layer": r.get("layer", ""),
                "alpha": r.get("alpha", ""),
            }
            for k in FIELDS:
                row[f"novice_base.{k}"] = m_nb[k]
                row[f"novice_plusv.{k}"] = m_np[k]
                row[f"expert_base.{k}"] = m_eb[k]
                row[f"expert_minusv.{k}"] = m_em[k]
                row[f"delta_novice_plusv_minus_base.{k}"] = m_np[k] - m_nb[k]
                row[f"delta_expert_minusv_minus_base.{k}"] = m_em[k] - m_eb[k]

                agg[f"novice+v Δ{k}"].append(m_np[k] - m_nb[k])
                agg[f"expert-v Δ{k}"].append(m_em[k] - m_eb[k])

            csv_rows.append(row)

    # Write CSV
    header = list(csv_rows[0].keys()) if csv_rows else []
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for row in csv_rows:
            w.writerow(row)

    print(f"Wrote scored JSONL -> {OUT_JSONL}")
    print(f"Wrote compare CSV  -> {OUT_CSV}\n")

    # Aggregate summary: directly answers your questions
    print("=== Aggregate mean deltas (steered - baseline) ===")
    print("Novice+v (should become MORE expert-like: harder / more technical):")
    for k in FIELDS:
        print(f"  Δ{k}: {mean(agg[f'novice+v Δ{k}']):+.3f}")

    print("\nExpert−v (should become MORE novice-like: easier / more definitions):")
    for k in FIELDS:
        print(f"  Δ{k}: {mean(agg[f'expert-v Δ{k}']):+.3f}")

if __name__ == "__main__":
    main()
