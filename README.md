# Diagnosing and Mitigating Object Hallucination in Vision-Language Models

## Task B — Inter IIT Tech Meet 15.0 Prepathon: Computer Vision

This repository contains the implementation, evaluation pipeline, experiments, and parameter-efficient mitigation strategy developed for **Task B: Diagnosing Object Hallucination in Vision-Language Models** under the **Inter IIT Tech Meet 15.0 Prepathon — Computer Vision Problem Statement**.

The project investigates whether a lightweight adaptation strategy can make a Vision-Language Model (VLM) rely more strongly on visual evidence and less on language or object co-occurrence priors when answering object-presence questions.

The final system uses **`HuggingFaceTB/SmolVLM-500M-Instruct`** and evaluates object hallucination using the **POPE benchmark**, with a particular focus on the adversarial split. The principal mitigation is **Low-Rank Adaptation (LoRA)** applied to the attention `q_proj` and `v_proj` modules.

> **Core result:** LoRA with rank `r = 16` reduces the adversarial False Positive Ratio (FPR) from **0.20 → 0.16**, a **20% relative reduction**, while maintaining **0.81 overall accuracy** and training only **0.4372%** of the model parameters.

---

## 1. Research Question

The project is centered around the following question:

> **Can a lightweight parameter-efficient adaptation strategy make a VLM rely more on image evidence than language and object co-occurrence priors, thereby reducing false-positive object claims on POPE without sacrificing balanced accuracy?**

The investigation compares three increasingly targeted interventions:

1. **Zero-shot baseline** — standard conversational prompting.
2. **Visual-verification prompting** — explicitly instructing the model to rely on visible evidence and answer `no` when the queried object cannot be clearly verified.
3. **LoRA fine-tuning** — adapting only selected attention projection layers while freezing the pretrained backbone.

---

## 2. Why Object Hallucination?

Object hallucination occurs when a VLM claims that an object is present even though the object is absent from the image.

The experiments and diagnostic analysis focus on four interacting mechanisms:

### Language Priors

The autoregressive language model learns strong statistical relationships between words, scenes, and objects. These priors can influence generation even when the image does not provide supporting evidence.

### Object Co-occurrence Bias

Objects that frequently appear together can create incorrect affirmative predictions. For example, a visible knife can increase the likelihood of predicting a fork even when no fork is present.

### Weak Visual Grounding

Visual representations are projected into the language-model representation space. When visual evidence is compressed or ambiguous, language-side attention patterns can dominate the decision.

### Generation Pressure

Instruction-following models can develop an affirmative confirmation tendency. This produces a measurable bias in the distribution of `yes` and `no` responses.

These failure modes motivate evaluating hallucination through multiple metrics instead of accuracy alone.

---

## 3. POPE Benchmark

The **Polling-based Object Probing Evaluation (POPE)** converts object hallucination into a controlled binary classification task.

Given an image `I` and an object `o`, the model receives a question such as:

```text
Is there a <object> in the image? Answer yes or no.
```

The model must produce:

```text
{yes, no}
```

POPE uses several negative-object sampling strategies:

| Split | Description |
|---|---|
| **Random** | Absent objects sampled randomly from non-present categories |
| **Popular** | Absent objects sampled using object-frequency statistics |
| **Adversarial** | Absent objects selected using strong co-occurrence relationships with objects present in the image |

This project uses the **Adversarial** split as the primary evaluation setting because it directly stresses language and co-occurrence priors.

---

## 4. Evaluation Metrics

Let `TP`, `TN`, `FP`, and `FN` denote true positives, true negatives, false positives, and false negatives.

### Overall Accuracy

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

Measures the fraction of all queries classified correctly.

### False Positive Ratio (FPR)

```text
FPR = FP / (FP + TN)
```

This is the primary hallucination metric. A false positive corresponds to claiming that an absent object is present.

### Yes/No Bias

```text
Yes Ratio = (TP + FP) / (TP + TN + FP + FN)
```

On a balanced evaluation set, a value near `0.50` indicates that the model is not strongly biased toward either response class.

### Precision

```text
Precision = TP / (TP + FP)
```

### Recall

```text
Recall = TP / (TP + FN)
```

### F1 Score

```text
F1 = 2 × Precision × Recall / (Precision + Recall)
```

A useful mitigation must reduce false positives without simply forcing the model to answer `no` to everything.

---

# 5. Model Backbone

The experiments use:

```text
HuggingFaceTB/SmolVLM-500M-Instruct
```

### Model characteristics

| Property | Configuration |
|---|---|
| Model | `HuggingFaceTB/SmolVLM-500M-Instruct` |
| Total parameters | **509,710,528** |
| Precision | **bfloat16** |
| Vision encoder | Compact multimodal vision encoder |
| Initial idle VRAM | **~1.2 GB** |

The sub-billion-parameter scale makes the model suitable for experimentation on a local laptop GPU while still exposing multimodal hallucination behavior under adversarial probing.

Upstream model page: 
**https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct**

---

# 6. Hardware and Software Environment

The entire experiment was developed under a strict consumer-hardware constraint.

| Component | Configuration |
|---|---|
| GPU | **NVIDIA GeForce RTX 4050 Laptop GPU** |
| Physical VRAM | **~6,141 MiB (~6 GB)** |
| CUDA | **13.0** |
| PyTorch | **2.14.0** |
| Python | **3.12** |
| OS | **Fedora Linux (x86-64)** |
| Precision | **bfloat16** |
| PEFT | **0.21.0** |

The 6 GB VRAM ceiling shaped the training architecture and required explicit memory optimization.

---

# 7. Memory-Efficient Training

Full-model fine-tuning is impractical under a 6 GB VRAM budget because activations, gradients, optimizer states, and model parameters all contribute to peak memory usage.

The final pipeline therefore uses parameter-efficient fine-tuning together with several memory optimizations.

## Gradient Checkpointing

```python
peft_model.gradient_checkpointing_enable()
```

Gradient checkpointing trades additional computation for lower activation memory by recomputing selected activations during backpropagation.

## Input Gradients

For PEFT/checkpointing compatibility:

```python
peft_model.enable_input_require_grads()
```

## Explicit Memory Cleanup

Before memory-intensive operations:

```python
gc.collect()
torch.cuda.empty_cache()
```

These operations reduce avoidable memory retention and allocator fragmentation.

---

# 8. Methodology

## 8.1 Baseline: Zero-Shot Probing

The baseline uses the standard conversational template:

```text
USER: <image>
{question}
Answer yes or no.
ASSISTANT:
```

Decoding uses greedy generation with:

```text
max_new_tokens = 5
```

The generated response is isolated from the prompt and normalized into a strict binary `yes` / `no` prediction.

This establishes the model's natural object-hallucination behavior before mitigation.

---

## 8.2 Visual-Verification Prompting

The second intervention adds an explicit instruction to prioritize visible evidence:

```text
USER: <image>
{question}
Look only at the image.
If the object is not clearly visible,
you must say no. Answer yes or no.
ASSISTANT:
```

### Intended effect

The prompt attempts to counter the model's affirmative confirmation prior without changing model parameters.

### Observed trade-off

The approach reduces FPR from:

```text
0.20 → 0.18
```

but also reduces inference throughput because of the longer prompt:

```text
3.65 it/s → 1.27 it/s
```

---

# 9. LoRA Fine-Tuning

The final mitigation uses **Low-Rank Adaptation (LoRA)**.

Instead of updating the full pretrained weight matrix `W₀`, LoRA introduces a low-rank update:

```text
W = W₀ + ΔW

ΔW = (α / r) BA
```

where:

```text
B ∈ R^(d × r)
A ∈ R^(r × k)
r << min(d, k)
```

The adapters are attached specifically to:

```text
q_proj
v_proj
```

### Final LoRA Configuration

| Parameter | Value |
|---|---:|
| Target modules | `q_proj`, `v_proj` |
| Rank `r` | **16** |
| Scaling `α` | **32** |
| Scaling factor `α/r` | **2.0** |
| Optimizer | AdamW |
| Learning rate | **2 × 10⁻⁴** |
| Batch size | **1** |
| Gradient accumulation | **1** |
| Precision | **bfloat16** |

Only **2,228,224** parameters are trainable:

```text
Trainable parameters : 2,228,224
Total parameters     : 509,710,528
Trainable ratio      : 0.4372%
```

More than **99.56%** of the pretrained model remains frozen.

---

# 10. Why `q_proj` and `v_proj`?

The attention query projection influences how representations form retrieval patterns, while the value projection controls information propagated through attention outputs.

Targeting these two projections provides a compact mechanism for adapting multimodal attention without updating the complete backbone.

This design is especially useful under the 6 GB VRAM constraint because the number of trainable parameters and optimizer states is dramatically smaller than in full fine-tuning.

The empirical result is a lower adversarial FPR without requiring full-model optimization.

---

# 11. Loss Masking

Training uses causal cross-entropy, but only the response tokens contribute to the objective.

```text
L(θ) = - Σ log Pθ(yt | y<t, Xv, Xq)
```

where the sum begins after the prompt tokens.

Prompt-token labels are assigned the ignore index:

```text
-100
```

This prevents the model from being penalized for reproducing the input prompt and concentrates the learning signal on the desired `yes` / `no` response.

---

# 12. Dataset Curation and Anti-Leakage

The evaluation uses **100 POPE Adversarial samples**.

Training data is taken exclusively from the POPE **Random** category to avoid direct evaluation leakage.

The final training set contains:

```text
100 positive samples
100 negative samples
--------------------
200 total samples
```

The two classes are concatenated and shuffled with seed `42`.

---

# 13. The 50/50 Class-Balance Trap

One of the most important findings in the project is that **strict class balance is essential**.

An initial experiment trained LoRA on an uncurated sample with a disproportionate number of positive examples. The adapter learned a shortcut: predicting the majority response class instead of improving visual grounding.

The failure produced:

```text
FPR       : 0.16 → 0.42
Yes Ratio : 0.47 → 0.70
Accuracy  : 0.81 → 0.78
```

This is a class-collapse failure.

### Remediation

The final training set enforces:

```text
50% Yes
50% No
```

or exactly:

```text
100 Yes + 100 No
```

This prevents answer-frequency shortcuts and forces the adapter to learn the distinction between visual presence and visual absence.

> **Key diagnostic lesson:** Hallucination mitigation must be evaluated under controlled class balance. Otherwise, apparent FPR changes can come from response-distribution collapse rather than improved visual grounding.

---

# 14. Quantitative Results

Evaluation was performed on the **100-sample POPE Adversarial split**.

| Metric | Baseline | Prompt Engineered | LoRA (Ours) |
|---|---:|---:|---:|
| **Overall Accuracy** | 0.8100 | 0.8200 | **0.8100** |
| **False Positive Ratio (FPR)** | 0.2000 | 0.1800 | **0.1600** |
| **Yes/No Ratio** | 0.5100 | 0.5000 | 0.4700 |
| **Precision** | 0.8039 | 0.8200 | **0.8298** |
| **Recall** | 0.8200 | 0.8200 | 0.7800 |
| **F1** | 0.8119 | 0.8200 | 0.8041 |
| **Relative FPR Reduction** | — | 10.0% | **20.0%** |
| **Trainable Parameters** | 0 | 0 | **2,228,224** |
| **Trainable Ratio** | 0.0% | 0.0% | **0.4372%** |
| **Inference Throughput** | **3.65 it/s** | 1.27 it/s | 1.21 it/s |

### Principal Result

The most important result is:

```text
Baseline FPR : 0.20
LoRA FPR     : 0.16
```

Relative reduction:

```text
(0.20 - 0.16) / 0.20 × 100 = 20.0%
```

The LoRA configuration therefore reduces false-positive object hallucination by **20% relative to the zero-shot baseline**.

---

# 15. Confusion-Matrix Interpretation

Because the evaluation set is balanced, the reported aggregate metrics correspond to the following confusion matrices.

### Baseline

```text
TP = 41
TN = 40
FP = 10
FN =  9
```

Resulting metrics:

```text
Precision = 0.8039
Recall    = 0.8200
F1        ≈ 0.8119
```

### Prompt Engineered

```text
TP = 41
TN = 41
FP =  9
FN =  9
```

Resulting metrics:

```text
Precision = 0.8200
Recall    = 0.8200
F1        = 0.8200
```

### LoRA

```text
TP = 39
TN = 42
FP =  8
FN = 11
```

Resulting metrics:

```text
Precision = 0.8298
Recall    = 0.7800
F1        ≈ 0.8041
```

This shows the central trade-off introduced by the mitigation: false positives decrease and precision improves, while recall decreases modestly.

---

# 16. Yes/No Bias Analysis

The affirmative response ratio changes as follows:

```text
Baseline   : 0.51
Prompting  : 0.50
LoRA       : 0.47
```

The LoRA model becomes somewhat more conservative about affirmative object claims, but does not collapse into predominantly negative answers.

This is importantly different from the unbalanced-training failure, where the model developed a strong affirmative bias of approximately `0.70`.

---

# 17. Training Dynamics

The balanced LoRA configuration showed stable convergence before early stopping.

| Step | Loss | Training State |
|---:|---:|---|
| 0 | 7.2730 | Initial forward pass / adapter initialization |
| 20 | 1.2473 | Rapid parameter alignment |
| 400 | 0.0862 | Stable convergence |
| 600 | 0.0766 | Reduction of false-confirmation logits |
| 800 | **0.0026** | Global minimum; best checkpoint saved |
| 1000 | 0.0048 | Generalization plateau |
| 1200 | 0.1440 | Patience limit reached; early stopping |

The best recorded loss was **0.0026**. The best adapter state was restored after early stopping.

---

# 18. PEFT State-Dictionary Engineering

A separate engineering failure occurred while saving checkpoints using the normal PyTorch mechanism:

```python
model.state_dict()
```

In PEFT / quantized environments, this can expose internal quantization-specific keys such as:

```text
.absmax
.quant_map
.quant_state.bitsandbytes__nf4
```

Attempting to restore such a complete state dictionary can result in namespace or key-mismatch errors.

The final solution isolates the adapter state using:

```python
get_peft_model_state_dict(peft_model)
```

and restores it using:

```python
set_peft_model_state_dict(peft_model, weights)
```

This approach:

- Saves only the trainable adapter parameters.
- Avoids unnecessary serialization of the frozen backbone.
- Prevents incompatibilities with quantized internal tensors.
- Makes best-checkpoint restoration reliable under constrained hardware.

---

# 19. Prompting vs. LoRA

The experiments reveal a useful distinction between instruction-only and parameter-level mitigation.

| Property | Prompt Engineering | LoRA |
|---|---|---|
| Parameter updates | None | **2.23M** |
| FPR | 0.18 | **0.16** |
| Overall accuracy | 0.82 | 0.81 |
| Yes/No ratio | 0.50 | 0.47 |
| Throughput | 1.27 it/s | 1.21 it/s |
| Persistent adaptation | No | **Yes** |
| Changes attention parameters | No | **Yes** |

Prompting can change the model's generation policy, but it does not directly alter the learned attention projections.

LoRA instead changes a small, targeted subset of the multimodal attention pathway while leaving the majority of the pretrained model untouched.

---

# 20. Alternative Mitigation — Visual Contrastive Decoding

**Visual Contrastive Decoding (VCD)** is an alternative inference-time strategy for mitigating hallucination without additional model training.

The method contrasts generation behavior for the original visual input with generation behavior under a visually distorted input. The goal is to suppress components of the output that remain strong despite unreliable visual evidence.

In this project, LoRA was selected as the final intervention because it provides a lightweight persistent adaptation while remaining practical within the 6 GB VRAM constraint.

---

# 21. Broader Evaluation Context

POPE is intentionally focused on binary object-presence hallucination. It is therefore highly useful for controlled diagnosis but does not cover every form of multimodal hallucination.

Relevant complementary resources include:

- **THRONE** — evaluates object hallucination in free-form VLM generations.
- **A-OKVQA** — evaluates visual question answering requiring broader world knowledge.
- **LLaVA** — an important reference for the visual instruction-tuning paradigm.

These benchmarks can be used in future work to determine whether improvements on binary POPE probing transfer to open-ended multimodal generation and knowledge-intensive visual reasoning.

---

# 22. Repository Structure

```text
.
├── data/
│   └── pope/
│       ├── random/
│       ├── popular/
│       └── adversarial/
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   └── 02_vlm_finetuning_v3.ipynb
│
├── reports/
│   ├── Task_B_Technical_Report.pdf
│   └── Task_B_Report_Source.tex
│
├── results/
│   └── evaluation_metrics.json
│
├── .gitignore
├── requirements.txt
└── README.md
```

### Key Files

#### `data/pope/`

Contains the POPE benchmark data used for training/evaluation experiments, organized by sampling strategy.

#### `notebooks/01_dataset_exploration.ipynb`

Used for inspecting the POPE dataset, class distribution, question structure, and parsing behavior.

#### `notebooks/02_vlm_finetuning_v3.ipynb`

Contains the primary end-to-end experiment:

```text
Model initialization
      ↓
Baseline evaluation
      ↓
Prompt-engineering evaluation
      ↓
Balanced dataset construction
      ↓
LoRA injection
      ↓
Masked causal-loss training
      ↓
Checkpoint restoration
      ↓
Final adversarial evaluation
```

#### `reports/`

Contains the formal technical report and its LaTeX source.

**[Task B Technical Report](./Reports/ReportTask_B_VLM.pdf)**

Stores evaluation outputs and metric information generated during experiments.

---

# 23. Quickstart

## Clone the Repository

```bash
git clone https://github.com/VirensinIITBHU/Diagnosing-Object-Hallucination-VLM.git
cd Diagnosing-Object-Hallucination-VLM
```

## Create the Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

The environment is designed around PyTorch, Transformers, PEFT, datasets, and supporting evaluation utilities.

---

# 24. Reproduction Workflow

Open:

```text
notebooks/02_vlm_finetuning_v3.ipynb
```

and execute the workflow sequentially.

### Stage 1 — Baseline

Initialize `SmolVLM-500M-Instruct` and evaluate its zero-shot response behavior on POPE Adversarial.

### Stage 2 — Prompt Intervention

Evaluate the visual-verification prompt and compare FPR, accuracy, Yes/No ratio, and throughput.

### Stage 3 — Balanced Training Set

Construct the 200-sample training pool:

```text
100 positive
100 negative
```

### Stage 4 — LoRA Injection

Configure:

```text
Target modules : q_proj, v_proj
Rank           : 16
Alpha          : 32
```

### Stage 5 — Training

Use masked causal cross-entropy, AdamW, and the memory-optimized PEFT configuration.

### Stage 6 — Checkpoint Restoration

Use:

```python
get_peft_model_state_dict()
set_peft_model_state_dict()
```

rather than serializing the complete PEFT / quantized model state.

### Stage 7 — Final Evaluation

Evaluate the restored adapter on the held-out **POPE Adversarial** set.

---

# 25. Final Results at a Glance

| Metric | Baseline | LoRA (Ours) |
|---|---:|---:|
| **False Positive Ratio** | 0.2000 | **0.1600** |
| **Relative FPR Reduction** | — | **20.0%** |
| **Overall Accuracy** | 0.8100 | **0.8100** |
| **Yes/No Ratio** | 0.5100 | **0.4700** |
| **Precision** | 0.8039 | **0.8298** |
| **Recall** | 0.8200 | 0.7800 |
| **F1** | 0.8119 | 0.8041 |
| **Trainable Parameters** | 0 | **2,228,224** |
| **Trainable Ratio** | 0.0% | **0.4372%** |

### Core Outcome

```text
              BEFORE                 AFTER

Baseline      FPR = 0.20            LoRA FPR = 0.16
Accuracy      0.81                  0.81
Yes Ratio     0.51                  0.47
Trainable     0%                    0.4372%
```

---

# 26. Technical Report

The complete methodology, mathematical formulation, experimental results, ablations, and failure analysis are documented in the technical report:

**[Task B Technical Report](./Reports/ReportTask_B_VLM.pdf)**

---

# 27. Video Demonstration

A short demonstration of the complete pipeline can be linked here:

**[Watch the Task B Demo](ADD_VIDEO_LINK_HERE)**

A useful demonstration can show:

```text
POPE image + question
        ↓
Zero-shot response
        ↓
Prompt-engineered response
        ↓
LoRA-adapted response
        ↓
Metric comparison
        ↓
Hallucination reduction
```

---

# 28. External Assets, Models, and References

## Model

**SmolVLM-500M-Instruct**  
Hugging Face: https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct

## POPE Benchmark

Li, Y., Du, Y., Zhou, K., Wang, J., Zhao, X., & Wen, J.-R. (2023).  
*Evaluating Object Hallucination in Large Vision-Language Models.*  
Proceedings of EMNLP 2023.  
DOI: https://doi.org/10.18653/v1/2023.emnlp-main.20

## LoRA

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022).  
*LoRA: Low-Rank Adaptation of Large Language Models.*  
International Conference on Learning Representations (ICLR).  
https://arxiv.org/abs/2106.09685

## LLaVA

Liu, H., Li, C., Wu, Q., & Lee, Y. J. (2023).  
*Visual Instruction Tuning.*  
https://arxiv.org/abs/2304.08485

## Visual Contrastive Decoding

Leng, S., Zhang, H., Chen, G., Li, X., Lu, S., Miao, C., & Bing, L. (2024).  
*Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding.*  
CVPR 2024, 13872–13882.  
https://doi.org/10.1109/CVPR52733.2024.01316

## THRONE

Kaul, P., Li, Z., Yang, H., Dukler, Y., Swaminathan, A., Taylor, C. J., & Soatto, S. (2024).  
*THRONE: An Object-Based Hallucination Benchmark for the Free-Form Generations of Large Vision-Language Models.*  
CVPR 2024, 27218–27228.  
https://doi.org/10.1109/CVPR52733.2024.02571

## A-OKVQA

Schwenk, D., Khandelwal, A., Clark, C., Marino, K., & Mottaghi, R. (2022).  
*A-OKVQA: A Benchmark for Visual Question Answering Using World Knowledge.*  
Computer Vision -- ECCV 2022, 146–162.  
https://doi.org/10.1007/978-3-031-20074-8_9

## Hallucination Survey

Liu, H., Xue, W., Chen, Y., Chen, D., Zhao, X., Wang, K., Hou, L., Li, R., & Peng, W. (2024).  
*A Survey on Hallucination in Large Vision-Language Models.*  
https://arxiv.org/abs/2402.00253

---

# 29. Collaborators

The project includes the following repository collaborators:

- `AMOGHA1140`
- `Abhyu-S`

---

# 30. Limitations

Several limitations should be considered when interpreting the reported results:

1. The primary quantitative evaluation uses a **100-sample POPE Adversarial subset**, so the reported numbers represent the experimental protocol used in this project rather than a comprehensive evaluation across all POPE samples.
2. POPE measures binary object-presence hallucination and does not fully capture hallucination in open-ended descriptions or long-form multimodal reasoning.
3. The reduction in FPR is accompanied by a decrease in Recall, indicating a precision-recall trade-off.
4. The training dataset is intentionally small and carefully balanced; broader data may be required to establish generalization beyond the evaluated setting.
5. The experiments are optimized for a **6 GB VRAM** environment, which constrains model size, training batch size, and experimentation breadth.

---

# 31. Key Takeaways

The final system demonstrates that object-hallucination mitigation can be approached as a controlled multimodal adaptation problem rather than solely as a prompt-engineering problem.

The main conclusions are:

- **LoRA reduces adversarial FPR from 0.20 to 0.16.**
- The reduction corresponds to a **20% relative decrease** in false-positive object claims.
- Only **0.4372% of model parameters** are trained.
- The final model retains **0.81 overall accuracy**.
- Strict **50/50 class balance** is essential to prevent class collapse.
- Gradient checkpointing and PEFT-specific state handling make training practical on **6 GB VRAM**.
- Prompting can reduce hallucination, but longer prompts introduce an observable throughput penalty.
- Lightweight attention adaptation provides a persistent mitigation mechanism without full-model fine-tuning.

---

## Project Summary

```text
Task                  : Object Hallucination Diagnosis
Benchmark             : POPE
Primary Evaluation    : POPE Adversarial
Backbone              : SmolVLM-500M-Instruct
Mitigation            : LoRA
Target Modules        : q_proj, v_proj
LoRA Rank             : 16
LoRA Alpha            : 32
Trainable Parameters  : 2,228,224
Trainable Ratio       : 0.4372%
Baseline FPR          : 0.20
Final FPR             : 0.16
Relative FPR Drop     : 20.0%
Final Accuracy        : 0.81
Final Yes/No Ratio    : 0.47
GPU                   : RTX 4050 Laptop
VRAM                  : ~6 GB
```
