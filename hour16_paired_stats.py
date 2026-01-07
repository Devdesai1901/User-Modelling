# hour16_paired_stats.py
import pandas as pd
from scipy.stats import wilcoxon

CSV_PATH = "hour14_compare.csv"

# (metric, expected_sign)
TESTS = {
    "novice+v": {
        "fk_grade": +1,
        "fk_ease": -1,
        "definition_count": -1,
        "jargon_rate": +1,
    },
    "expert-v": {
        "fk_grade": -1,
        "fk_ease": +1,
        "definition_count": +1,
        "jargon_rate": -1,
    }
}

def run_test(series, expected_sign):
    deltas = series.dropna()
    if len(deltas) < 5:
        return None

    # Wilcoxon signed-rank test vs zero
    stat, p = wilcoxon(deltas)

    mean = deltas.mean()
    median = deltas.median()

    # Fraction moving in expected direction
    frac_correct = (expected_sign * deltas > 0).mean()

    return {
        "n": len(deltas),
        "mean_delta": mean,
        "median_delta": median,
        "frac_correct_direction": frac_correct,
        "p_value": p,
    }

def main():
    df = pd.read_csv(CSV_PATH)

    print(f"Loaded {len(df)} paired examples\n")

    for condition, metrics in TESTS.items():
        print(f"=== {condition.upper()} (paired Wilcoxon) ===")
        for metric, sign in metrics.items():
            col = f"delta_{condition.replace('+','_plus').replace('-','_minus')}_minus_base.{metric}"
            if col not in df.columns:
                print(f"  {metric}: column missing")
                continue

            res = run_test(df[col], sign)
            if res is None:
                print(f"  {metric}: insufficient data")
                continue

            print(
                f"  {metric:18s} "
                f"mean={res['mean_delta']:+.3f} "
                f"median={res['median_delta']:+.3f} "
                f"frac_dir={res['frac_correct_direction']:.2f} "
                f"p={res['p_value']:.3g}"
            )
        print()

if __name__ == "__main__":
    main()
