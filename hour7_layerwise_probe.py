import json
import os
import random
import numpy as np
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

JSONL_PATH = "hour5_baseline.jsonl"   # your index file
OUT_DIR = "hour7_probe_out"
SEED = 42

# Use a smaller N first for quick debugging; set to None for full
MAX_ROWS = None  # e.g., 200 for quick test, None for full

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_rows(path: str, max_rows=None):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            rows.append(json.loads(line))
    return rows

def main():
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    rows = load_rows(JSONL_PATH, MAX_ROWS)
    if len(rows) == 0:
        raise RuntimeError(f"No rows found in {JSONL_PATH}")

    # Labels: novice=0, expert=1
    y = np.array([1 if r["condition"] == "expert" else 0 for r in rows], dtype=np.int64)

    # Load first act tensor to get L,d
    X0 = torch.load(rows[0]["act_path"], map_location="cpu")
    assert X0.ndim == 2, f"Expected (L,d) tensor, got {tuple(X0.shape)}"
    L, d = X0.shape

    # Preload all activations into RAM for simplicity (works for 1000 rows × 28 × 3584 ~ a few hundred MB)
    # If RAM becomes an issue, we can stream per-layer later.
    acts = []
    for r in rows:
        X = torch.load(r["act_path"], map_location="cpu")  # (L,d)
        if X.shape != (L, d):
            raise RuntimeError(f"Bad shape in {r['act_path']}: {tuple(X.shape)} expected {(L,d)}")
        acts.append(X.numpy().astype(np.float32))
    acts = np.stack(acts, axis=0)  # (N, L, d)

    N = acts.shape[0]
    print(f"Loaded N={N} examples, L={L} layers, d={d}")

    # Train/test split (stratified)
    idx = np.arange(N)
    train_idx, test_idx = train_test_split(
        idx, test_size=0.2, random_state=SEED, stratify=y
    )

    results = []
    best = None  # (metric, layer, clf)

    # Baseline: majority class accuracy
    majority = int(np.round(y[train_idx].mean()))  # 0 or 1
    baseline_acc = (y[test_idx] == majority).mean()
    print(f"Baseline (majority) test accuracy: {baseline_acc:.3f}")

    for l in range(L):
        X_train = acts[train_idx, l, :]  # (Ntrain, d)
        X_test  = acts[test_idx,  l, :]  # (Ntest, d)

        clf = LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            C=1.0,
            random_state=SEED
        )
        clf.fit(X_train, y[train_idx])

        probs = clf.predict_proba(X_test)[:, 1]
        pred = (probs >= 0.5).astype(np.int64)

        acc = accuracy_score(y[test_idx], pred)
        try:
            auc = roc_auc_score(y[test_idx], probs)
        except ValueError:
            auc = float("nan")

        results.append({"layer": l, "acc": float(acc), "auc": float(auc)})

        # Choose best by AUC primarily; fallback to acc if AUC nan
        score = auc if not np.isnan(auc) else acc
        if best is None or score > best[0]:
            best = (score, l, clf)

        if l % max(1, L // 10) == 0:
            print(f"Layer {l:02d}: acc={acc:.3f} auc={auc:.3f}")

    # Save results
    results_path = os.path.join(OUT_DIR, "layerwise_probe_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "N": int(N),
                "L": int(L),
                "d": int(d),
                "baseline_acc": float(baseline_acc),
                "results": results,
                "best_layer": int(best[1]),
                "best_score": float(best[0]),
            },
            f,
            indent=2
        )

    # Save best probe weights
    best_layer = best[1]
    best_clf = best[2]
    w = best_clf.coef_.reshape(-1).astype(np.float32)  # (d,)
    b = float(best_clf.intercept_[0])

    np.save(os.path.join(OUT_DIR, f"best_probe_w_layer{best_layer}.npy"), w)
    with open(os.path.join(OUT_DIR, f"best_probe_b_layer{best_layer}.txt"), "w") as f:
        f.write(str(b))

    print("\n=== Best layer ===")
    print(f"Layer: {best_layer}")
    print(f"Score (AUC or acc): {best[0]:.3f}")
    print(f"Saved: {results_path}")
    print(f"Saved probe weights to: {OUT_DIR}/best_probe_w_layer{best_layer}.npy")

if __name__ == "__main__":
    main()
