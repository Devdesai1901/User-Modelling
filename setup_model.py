import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Force everything onto GPU and use bf16 (much more stable than fp16)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map={"": "cuda:0"},   # <-- important: avoid CPU offload
    trust_remote_code=True
).eval()

messages = [{"role": "user", "content": "Explain what a neural network is."}]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
    )

# Decode ONLY the generated continuation (not the prompt)
gen_ids = out[0, inputs["input_ids"].shape[1]:]
print(tokenizer.decode(gen_ids, skip_special_tokens=True))
