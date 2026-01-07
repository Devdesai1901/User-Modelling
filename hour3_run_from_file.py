# hour5_generate_and_save_acts.py
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hooks import ResidualStreamCaptureLastTok
from metrics import explanation_metrics

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = "cuda:3"

IN_PATH = "generated_prompts.json"
OUT_PATH = "hour5_baseline.jsonl"
N_USE = 500  # set to 500 when ready

# Keep outputs comparable across runs
GEN_KWARGS = dict(
    max_new_tokens=200,
    do_sample=False,
)

def to_chat_messages(user_text: str):
    return [{"role": "user", "content": user_text}]

def run_one(model, tokenizer, capture, user_text: str):
    """
    Generate answer and populate capture.cache with last-token residual vectors
    at each layer. Returns answer, metrics, and prompt length.
    """
    capture.clear()

    messages = to_chat_messages(user_text)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        out = model.generate(**inputs, **GEN_KWARGS, eos_token_id=tokenizer.eos_token_id)

    gen_ids = out[0, inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(gen_ids, skip_special_tokens=True)

    return answer, explanation_metrics(answer), int(inputs["input_ids"].shape[1])

def main():
    # Load prompts
    with open(IN_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    data = data[:N_USE]

    # Create activation directory once
    os.makedirs("acts", exist_ok=True)

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map={"": DEVICE},
        trust_remote_code=True
    ).eval()

    # Hooks: capture only last-token vectors per layer
    capture = ResidualStreamCaptureLastTok()
    capture.add_hooks(model)

    rows = 0
    with open(OUT_PATH, "w", encoding="utf-8") as out_f:
        for i, ex in enumerate(data):
            topic = ex.get("topic", "")
            domain = ex.get("domain", "")
            novice_prompt = ex["novice_prompt"]
            expert_prompt = ex["expert_prompt"]

            for condition, user_text in [("novice", novice_prompt), ("expert", expert_prompt)]:
                answer, metrics, prompt_len = run_one(model, tokenizer, capture, user_text)

                # Save activations: stack into (L, d) tensor
                L = len(capture.cache)
                # Ensure deterministic ordering 0..L-1
                X = torch.stack([capture.cache[l] for l in range(L)], dim=0)  # (L, d), fp32 on CPU

                act_path = f"acts/idx{i}_{condition}.pt"
                torch.save(X, act_path)

                # (Optional) sanity checks on first 2 rows
                if rows < 2:
                    assert X.ndim == 2, f"Expected 2D tensor (L,d), got shape {tuple(X.shape)}"
                    print(f"[sanity] act tensor shape: {tuple(X.shape)} saved to {act_path}")

                row = {
                    "idx": i,
                    "topic": topic,
                    "domain": domain,
                    "condition": condition,
                    "user_prompt": user_text,
                    "answer": answer,
                    "metrics": metrics,
                    "prompt_len": prompt_len,
                    "act_path": act_path,
                    "n_layers": int(X.shape[0]),
                    "hidden_dim": int(X.shape[1]),
                }

                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows += 1
                print(f"[{rows:02d}] idx={i} {condition} | metrics={metrics} | act={act_path}")

    capture.remove_hooks()
    print(f"\nWrote {rows} rows ({N_USE} prompts × 2 conditions) to {OUT_PATH}")
    print("Activations saved under ./acts as .pt tensors of shape (n_layers, hidden_dim).")

if __name__ == "__main__":
    main()
