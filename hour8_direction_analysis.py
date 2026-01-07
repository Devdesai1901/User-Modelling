import json
import os
import numpy as np
import torch

JSONL_PATH = "hour5_baseline.jsonl"
BEST_LAYER = 20
PROBE_W_PATH = "hour7_probe_out/best_probe_w_layer20.npy"
OUT_DIR = "hour8_out"

os.makedirs(OUT_DIR, exist_ok=True)

def cosine(a, b, eps=1e-9):
    a = a / (np.linalg.norm(a) + eps)
    b = b / (np.linalg.norm(b) + eps)
    return float(np.dot(a, b))

def load_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def main():
    rows = load_rows(JSONL_PATH)
    expert, novice = [], []

    for r in rows:
        X = torch.load(r["act_path"], map_location="cpu")  # (L,d)
        x = X[BEST_LAYER].numpy().astype(np.float32)       # (d,)
        if r["condition"] == "expert":
            expert.append(x)
        else:
            novice.append(x)

    expert = np.stack(expert, axis=0)
    novice = np.stack(novice, axis=0)

    mu_e = expert.mean(axis=0)
    mu_n = novice.mean(axis=0)

    v = mu_e - mu_n
    v_unit = v / (np.linalg.norm(v) + 1e-9)

    # Probe direction
    w = np.load(PROBE_W_PATH).astype(np.float32)
    w_unit = w / (np.linalg.norm(w) + 1e-9)

    cos_sim = cosine(v_unit, w_unit)

    # Separation along v
    proj_e = expert @ v_unit
    proj_n = novice @ v_unit
    pooled_std = 0.5 * (proj_e.std() + proj_n.std()) + 1e-9
    sep = float((proj_e.mean() - proj_n.mean()) / pooled_std)

    # Save artifacts
    np.save(os.path.join(OUT_DIR, f"mu_expert_layer{BEST_LAYER}.npy"), mu_e)
    np.save(os.path.join(OUT_DIR, f"mu_novice_layer{BEST_LAYER}.npy"), mu_n)
    np.save(os.path.join(OUT_DIR, f"v_dom_layer{BEST_LAYER}.npy"), v_unit)

    summary = {
        "layer": BEST_LAYER,
        "n_expert": int(expert.shape[0]),
        "n_novice": int(novice.shape[0]),
        "d": int(expert.shape[1]),
        "cos_v_dom_vs_probe_w": float(cos_sim),
        "proj_mean_expert": float(proj_e.mean()),
        "proj_mean_novice": float(proj_n.mean()),
        "proj_std_expert": float(proj_e.std()),
        "proj_std_novice": float(proj_n.std()),
        "separation_means_over_pooled_std": float(sep),
    }

    with open(os.path.join(OUT_DIR, "direction_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=== Hour 8: Direction Analysis ===")
    print(f"Layer: {BEST_LAYER} | N expert={expert.shape[0]} N novice={novice.shape[0]} d={expert.shape[1]}")
    print(f"cos(v_dom, w_probe): {cos_sim:.3f}")
    print(f"proj mean expert={proj_e.mean():.3f} (std={proj_e.std():.3f})")
    print(f"proj mean novice={proj_n.mean():.3f} (std={proj_n.std():.3f})")
    print(f"separation (means / pooled_std): {sep:.3f}")
    print(f"Saved v_dom -> {OUT_DIR}/v_dom_layer{BEST_LAYER}.npy")
    print(f"Saved summary -> {OUT_DIR}/direction_summary.json")

if __name__ == "__main__":
    main()
