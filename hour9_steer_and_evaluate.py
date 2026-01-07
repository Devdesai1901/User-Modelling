import json
import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from metrics import explanation_metrics
from steering import ResidualSteererLastTok

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = "cuda:3"

LAYER = 20
V_PATH = "hour8_out/v_dom_layer20.npy"

IN_PATH = "generated_prompts.json"
N_USE = 30          # small eval set for hour 9; bump later
OUT_PATH = "hour9_steering_results.jsonl"

GEN_KWARGS = dict(
    max_new_tokens=220,
    do_sample=False,
)

# Try a small sweep. If effect is weak, increase alphas.
ALPHAS = [0.0, 0.5, 1.0, 2.0]  # we’ll run +alpha and -alpha (except 0)

def to_chat_messages(user_text: str):
    return [{"role": "user", "content": user_text}]

def generate(model, tokenizer, user_text: str):
    messages = to_chat_messages(user_text)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        out = model.generate(**inputs, **GEN_KWARGS, eos_token_id=tokenizer.eos_token_id)

    gen_ids = out[0, inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return answer

def main():
    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)

    # Load prompts
    with open(IN_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)[:N_USE]

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map={"": DEVICE},
        trust_remote_code=True
    ).eval()

    # Load v_dom and move to GPU
    v = np.load(V_PATH).astype(np.float32)             # unit vector (d,)
    v_t = torch.tensor(v, device=DEVICE, dtype=torch.bfloat16)

    # Output
    rows_written = 0
    with open(OUT_PATH, "w", encoding="utf-8") as out_f:
        for i, ex in enumerate(data):
            topic = ex.get("topic", "")
            domain = ex.get("domain", "")

            # Use a "neutral" prompt to test causality:
            # pick the novice_prompt (simple wording) but we will steer internally.
            user_text = f"Explain {ex['topic']}."


            # Baseline (alpha=0)
            base_answer = generate(model, tokenizer, user_text)
            base_metrics = explanation_metrics(base_answer)

            # Save baseline row
            out_f.write(json.dumps({
                "idx": i,
                "topic": topic,
                "domain": domain,
                "user_prompt": user_text,
                "alpha": 0.0,
                "sign": "none",
                "answer": base_answer,
                "metrics": base_metrics,
            }, ensure_ascii=False) + "\n")
            rows_written += 1

            # Steering runs
            for a in ALPHAS:
                if a == 0.0:
                    continue

                for sign in ["plus", "minus"]:
                    alpha = a if sign == "plus" else -a
                    steerer = ResidualSteererLastTok(LAYER, v_t, alpha)
                    steerer.add(model)

                    ans = generate(model, tokenizer, user_text)
                    met = explanation_metrics(ans)

                    steerer.remove()

                    out_f.write(json.dumps({
                        "idx": i,
                        "topic": topic,
                        "domain": domain,
                        "user_prompt": user_text,
                        "alpha": float(a),
                        "sign": sign,
                        "effective_alpha": float(alpha),
                        "answer": ans,
                        "metrics": met,
                    }, ensure_ascii=False) + "\n")
                    rows_written += 1

            print(f"Done idx={i} | topic={topic}")

    print(f"\nWrote {rows_written} rows to {OUT_PATH}")
    print("Now run the summary script to see metric shifts.")

if __name__ == "__main__":
    main()
