# hour8_proj_hist.py
import json, os, numpy as np, torch, matplotlib.pyplot as plt

JSONL_PATH="hour5_baseline.jsonl"
LAYER=20
V_PATH="hour8_out/v_dom_layer20.npy"
OUT="hour8_out/proj_hist_layer20.png"

rows=[json.loads(l) for l in open(JSONL_PATH,"r",encoding="utf-8")]
v=np.load(V_PATH).astype(np.float32)
ve, vn = [], []
for r in rows:
    X=torch.load(r["act_path"], map_location="cpu")
    p=float(X[LAYER].numpy().astype(np.float32) @ v)
    (ve if r["condition"]=="expert" else vn).append(p)

plt.figure()
plt.hist(vn, bins=30, alpha=0.7, label="novice")
plt.hist(ve, bins=30, alpha=0.7, label="expert")
plt.legend()
plt.title(f"Projection onto v_dom (layer {LAYER})")
plt.xlabel("x · v_dom")
plt.ylabel("count")
plt.savefig(OUT, dpi=200, bbox_inches="tight")
print("Saved:", OUT)
