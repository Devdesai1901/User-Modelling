# hour10_score_generations.py
import json
import csv
import argparse
from typing import Dict, Any, List

from metrics import explanation_metrics

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_csv(path: str, rows: List[Dict[str, Any]], metric_key="metrics_v2"):
    # Flatten metrics into columns
    fieldnames = sorted({k for r in rows for k in r.keys() if k != metric_key})
    metric_fields = sorted({mk for r in rows for mk in r.get(metric_key, {}).keys()})
    header = fieldnames + [f"{metric_key}.{m}" for m in metric_fields]

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in fieldnames}
            mets = r.get(metric_key, {})
            for m in metric_fields:
                out[f"{metric_key}.{m}"] = mets.get(m, "")
            w.writerow(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", type=str, required=True, help="Input JSONL (must contain 'answer').")
    ap.add_argument("--out_jsonl", type=str, required=True, help="Output JSONL with metrics_v2.")
    ap.add_argument("--out_csv", type=str, default="", help="Optional CSV output.")
    args = ap.parse_args()

    rows = read_jsonl(args.in_path)
    if not rows:
        raise RuntimeError(f"No rows read from {args.in_path}")

    # Score
    for r in rows:
        ans = r.get("answer", "")
        r["metrics_v2"] = explanation_metrics(ans)

    write_jsonl(args.out_jsonl, rows)

    if args.out_csv:
        write_csv(args.out_csv, rows)

    # Quick sanity print
    print(f"Scored {len(rows)} rows.")
    print("Example metrics_v2:", rows[0]["metrics_v2"])

if __name__ == "__main__":
    main()
