# Do Language Models Represent an Internal Estimate of User Knowledge?

This repository contains code and experiments investigating whether large language models (LLMs)
internally represent an estimate of a user’s knowledge level (e.g., *novice* vs *expert*), and whether
this representation causally influences explanation depth and complexity.

We study this question using **mechanistic interpretability techniques**: activation probing,
representation geometry analysis, and causal activation steering.

---

## 🧠 Research Question

Do LLMs implicitly track a user’s expertise and use it to tailor explanations?

More formally:
- Does a hidden “user knowledge” variable exist in the model’s internal activations?
- Is this variable linearly decodable?
- Does intervening on it causally change explanation style?

---

## 🔍 Key Idea

Humans naturally adapt explanations to a learner’s background.
LLMs can be *prompted* to do the same—but it is unclear whether they internally model user knowledge
or merely follow surface-level instructions.

This project probes whether **user expertise is represented as a geometric direction in hidden space**
and whether that direction functions as a **conditioning signal** during generation.

---

## 🧪 Experimental Setup (High-level)

### Model
- **Qwen2.5-7B-Instruct** (open-weights, chat-style)

### Prompt Design
- Synthetic Q&A prompts across ~20–30 topics
- Each question appears in two framings:
  - **Novice**: “I have no background, explain from basics.”
  - **Expert**: “As you know, given my PhD…, explain in depth.”

### Activation Capture
- Single forward pass per prompt
- Capture **residual stream activations** at every transformer layer
- Use the final prompt token (pre-generation)

---

## 📐 Representation Analysis

### Linear Probing
- Train layerwise logistic regression probes:
  - Input: residual activations
  - Target: novice vs expert label
- Probe performance peaks at **layer 20**
  - Accuracy ≈ **0.80**
  - AUC ≈ **0.89**

### Direction Extraction
At the best layer ℓ: v = μ_expert − μ_novice

This vector serves as a candidate **user-knowledge axis**.

---

## 🎛️ Causal Steering

During generation, we intervene on the residual stream:

h_new = h_orig ± α v


- α ∈ {0.5, 1.0, 2.0}
- Interventions applied at layer 20
- Compare steered vs baseline outputs **within the same prompt**

---

## 📊 Evaluation Metrics

We use behavioral proxies for explanation depth:
- Word count
- Flesch–Kincaid grade level
- Flesch reading ease
- Jargon rate
- Explicit definition count (“X is…”, “X means…”)

---

## 📈 Key Findings

- **Representation evidence**:  
  User knowledge is linearly decodable in mid-to-late layers, with a clear geometric separation.

- **Causal effects (modest but directional)**:
  - Adding +v to novice prompts slightly increases technicality
  - Removing v from expert prompts produces asymmetric, sometimes counterintuitive effects

- **Interpretation**:
  The user-knowledge signal acts as a **soft conditioning bias**, not a hard control knob.
  Explanation depth is determined downstream by planning and decoding circuits.

---

## ⚠️ Limitations

- Steering effects are small in magnitude
- Most shifts are not statistically significant
- Automated readability metrics are noisy proxies
- Effects are asymmetric: removing “expert” signal does not reliably simplify explanations

These results suggest partial control, not a reversible switch.

---

## 📁 Repository Structure

.
├── build_eli5_pairs.py # Prompt generation (novice/expert pairs)
├── setup_model.py # Model loading and configuration
├── hooks.py # Activation capture hooks
├── steering.py # Activation steering utilities
├── metrics.py # Readability and style metrics
├── hour_.py # Experiment stages (probing, steering, evaluation)
├── acts/ # (ignored) activation artifacts
└── .gitignore



> ⚠️ JSON/CSV outputs and activation dumps are intentionally excluded from version control.

---

## 🚀 Reproducing Experiments

1. Set environment variable:
```bash
export OPENAI_API_KEY=your_key_here
2. Generate prompts:

python build_eli5_pairs.py
3. Run probing and direction extraction:
python hour7_layerwise_probe.py
4. Apply steering and evaluate:
python hour9_steer_and_evaluate.py

