# hour17_controls_generate.py
import json
import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from steering import ResidualSteererLastTok  # your existing steering class

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = "cuda:3"

IN_PATH = "generated_prompts.json"
OUT_PATH = "hour17_controls.jsonl"

LAYER = 20
ALPHA = 1.0
V_PATH = "hour8_out/v_dom_layer20.npy"

N_USE = 30  # set to len(data) when ready

GEN_KWARGS = dict(
    max_new_tokens=150,
    do_sample=False,
)

def to_chat_prompt(tokenizer, user_text: str) -> str:
    messages = [{"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def generate(model, tokenizer, user_text: str) -> str:
    prompt = to_chat_prompt(tokenizer, user_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, **GEN_KWARGS, eos_token_id=tokenizer.eos_token_id)
    gen_ids = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)

def gen_with_vec(model, tokenizer, user_text: str, vec_t: torch.Tensor, alpha: float) -> str:
    steerer = ResidualSteererLastTok(LAYER, vec_t, alpha)
    steerer.add(model)
    try:
        return generate(model, tokenizer, user_text)
    finally:
        steerer.remove()

def main():
    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)

    data = json.load(open(IN_PATH, "r", encoding="utf-8"))[:N_USE]
    print(f"Loaded {len(data)} prompt pairs")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map={"": DEVICE},
        trust_remote_code=True
    ).eval()

    # Load v
    v = np.load(V_PATH).astype(np.float32)
    v_t = torch.tensor(v, device=DEVICE, dtype=torch.bfloat16)

    # Random direction r with same L2 norm as v (v is likely unit; still match norm exactly)
    rng = np.random.default_rng(0)
    r = rng.standard_normal(size=v.shape).astype(np.float32)
    r = r / (np.linalg.norm(r) + 1e-12) * (np.linalg.norm(v) + 1e-12)
    r_t = torch.tensor(r, device=DEVICE, dtype=torch.bfloat16)

    # Zero vector
    z_t = torch.zeros_like(v_t)

    rows = 0
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for i, ex in enumerate(data):
            topic = ex.get("topic", "")
            domain = ex.get("domain", "")
            novice_prompt = ex["novice_prompt"]
            expert_prompt = ex["expert_prompt"]

            # Baselines
            novice_base = generate(model, tokenizer, novice_prompt)
            expert_base = generate(model, tokenizer, expert_prompt)

            # Sanity / controls on novice
            novice_plusv  = gen_with_vec(model, tokenizer, novice_prompt, v_t, +ALPHA)
            novice_minusv = gen_with_vec(model, tokenizer, novice_prompt, v_t, -ALPHA)  # flip
            novice_rand   = gen_with_vec(model, tokenizer, novice_prompt, r_t, +ALPHA)  # random
            novice_zero   = gen_with_vec(model, tokenizer, novice_prompt, z_t, +ALPHA)  # zero

            # Controls on expert (optional but recommended)
            expert_minusv = gen_with_vec(model, tokenizer, expert_prompt, v_t, -ALPHA)
            expert_rand   = gen_with_vec(model, tokenizer, expert_prompt, r_t, +ALPHA)
            expert_zero   = gen_with_vec(model, tokenizer, expert_prompt, z_t, +ALPHA)

            row = {
                "idx": i,
                "topic": topic,
                "domain": domain,
                "layer": LAYER,
                "alpha": ALPHA,

                "novice_prompt": novice_prompt,
                "expert_prompt": expert_prompt,

                "novice_base": novice_base,
                "novice_plusv": novice_plusv,
                "novice_minusv": novice_minusv,
                "novice_rand": novice_rand,
                "novice_zero": novice_zero,

                "expert_base": expert_base,
                "expert_minusv": expert_minusv,
                "expert_rand": expert_rand,
                "expert_zero": expert_zero,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows += 1

            if (i + 1) % 10 == 0:
                print(f"[{i+1}/{len(data)}] wrote {rows} rows")

    print(f"Done. Wrote {rows} rows to {OUT_PATH}")

if __name__ == "__main__":
    main()
