# Does the Language You Ask In Change What the Model Sees?

## Decision-Point Language Information Leakage (DPLIL) in Vision-Language Models

This repository contains the code for investigating how query language modulates visual judgment in multilingual Vision-Language Models (VLMs). We discover that asking the same visual question in different languages causes VLMs to give semantically different answers — not just differently phrased responses, but fundamentally different visual judgments.

### Core Findings

- **Cross-lingual answer divergence is severe**: The accuracy gap between English and non-English queries reaches 34–40% across four VLMs, with only 12–16% of instances yielding consistent answers across all languages.
- **Language information leaks into the decision point**: Linear probes achieve ~100% accuracy in decoding query language from the answer-planning hidden state, while visual token positions carry no language signal.
- **The leakage is low-rank and localized**: Language information concentrates in 2–5 PCA dimensions and amplifies within a specific "language routing zone" of decoder layers.
- **Mean-shift correction is effective**: A simple, zero-cost intervention that shifts non-English representations toward the English mean reduces the gap by 41–82% across four architectures.
- **Causal verification**: SAE-based feature analysis confirms that Language-Modulated Features (LMF) are causally responsible — replacing LMF activations with English references outperforms random and language-invariant feature replacement.

---

## Method Overview

We formalize the problem via an information-theoretic invariance condition:

> **I(Â; L | I, m) = 0** — the model's answer semantics Â should carry zero information about the query language L, given the image I and question meaning m.

From this violated ideal, we derive three testable predictions:

1. **Decodability**: Language identity is linearly decodable from the answer-planning state
2. **Routing Concentration**: A contiguous layer interval amplifies language information into decision-relevant divergence
3. **Low-Rank Separability**: Language information occupies ≤5 dimensions of the full hidden space

We then propose **Mean-Shift Correction** — a geometry-informed repair that subtracts the per-language mean offset at the identified routing zone — and validate it with **SAE-based causal analysis** that decomposes features into Language-Modulated (LMF) and Language-Invariant (LIVF) categories.

---

## Repository Structure

```
code/
├── config.py                  # Shared config, answer normalization, prompt templates
├── sae_model.py               # Sparse Autoencoder architecture
│
├── prepare_xgqa.py            # Step 1: Build multilingual VQA dataset from xGQA + GQA
├── collect_last_token.py      # Step 2: Collect last-token hidden states across layers
│
├── train_sae.py               # Step 3: Train SAE on collected hidden states
│
├── pre_experiment.py          # Exp 1: Baseline cross-lingual consistency evaluation
├── diagnose_positions.py      # Exp 2a: Position diagnostic (visual vs. last token)
├── probe_and_diagnostics.py   # Exp 2b: Language probe + phi scores (LMF ranking)
├── probe_pca.py               # Exp 2c: PCA dimensionality analysis
├── analyze_crosslingual.py    # Exp 2d: SAE feature classification (LMF vs. LIVF)
│
├── nonoracle_repair.py        # Exp 3: Mean-shift correction + INLP baseline
├── cross_model_repair.py      # Exp 3: Cross-architecture validation
│
├── align_experiment.py        # Exp 4: Causal alignment (replace LMF with EN values)
├── dose_response.py           # Exp 4: Dose-response sweep (LMF vs. Random vs. LIVF)
├── control_experiments.py     # Exp 5: Controls (paraphrase, no-image, shuffled-image)
│
└── README.md
```

---

## Pipeline

The experimental pipeline consists of six phases:

### Phase 1: Data Preparation

```bash
python prepare_xgqa.py \
  --xgqa_dir <path_to_xgqa_zero_shot> \
  --image_dir <path_to_gqa_images> \
  --output_dir <output_dir> \
  --langs en,zh,ko,de
```

### Phase 2: Phenomenon Confirmation (Baseline)

```bash
python pre_experiment.py \
  --model_path <vlm_checkpoint> \
  --data_path <dataset_json> \
  --save_dir <results_dir> \
  --langs en,zh,ko,de
```

### Phase 3: Hidden State Collection

```bash
python collect_last_token.py \
  --model_path <vlm_checkpoint> \
  --model_type qwen \
  --data_path <dataset_json> \
  --save_dir <states_dir> \
  --layers <comma_separated_layer_indices> \
  --langs en,zh,ko,de
```

### Phase 4: SAE Training

```bash
python train_sae.py \
  --data_path <combined_states_pt> \
  --save_dir <checkpoint_dir> \
  --input_dim <hidden_size> \
  --hidden_dim <sae_dict_size> \
  --l1_coeff <sparsity_weight> \
  --lr <learning_rate> \
  --batch_size <batch_size> \
  --num_epochs <epochs>
```

### Phase 5: Mechanism Diagnosis

```bash
# Position diagnostic
python diagnose_positions.py \
  --model_path <vlm_checkpoint> \
  --data_path <dataset_json> \
  --save_path <output_json> \
  --image_token_id <token_id> \
  --layers <layer_indices>

# Language probes + phi scores
python probe_and_diagnostics.py \
  --states_dir <states_dir> \
  --sae_path <sae_checkpoint> \
  --output_dir <probe_output> \
  --layers <layer_indices>

# PCA analysis
python probe_pca.py \
  --states_dir <states_dir> \
  --output_dir <pca_output> \
  --layers <layer_indices>

# Feature classification
python analyze_crosslingual.py \
  --states_dir <states_dir> \
  --sae_path <sae_checkpoint> \
  --output_dir <analysis_output> \
  --layers <layer_indices>
```

### Phase 6: Repair & Causal Verification

```bash
# Mean-shift correction (non-oracle)
python nonoracle_repair.py \
  --model_path <vlm_checkpoint> \
  --data_path <dataset_json> \
  --states_dir <states_dir> \
  --save_dir <repair_output> \
  --method mean_shift \
  --layers <intervention_layers>

# Cross-model validation
python cross_model_repair.py \
  --model_path <vlm_checkpoint> \
  --model_type <qwen|llava|internvl> \
  --data_path <dataset_json> \
  --save_dir <repair_output> \
  --layers <intervention_layers>

# Causal alignment
python align_experiment.py \
  --model_path <vlm_checkpoint> \
  --data_path <dataset_json> \
  --sae_path <sae_checkpoint> \
  --analysis_path <crosslingual_analysis_pt> \
  --save_dir <alignment_output> \
  --condition align_bot20 \
  --intervention_layer <layer_idx>

# Dose-response
python dose_response.py \
  --model_path <vlm_checkpoint> \
  --data_path <dataset_json> \
  --sae_path <sae_checkpoint> \
  --phi_dir <probe_output> \
  --save_dir <dose_output> \
  --layers <layer_indices>

# Control experiments
python control_experiments.py \
  --model_path <vlm_checkpoint> \
  --data_path <dataset_json> \
  --save_dir <control_output> \
  --experiment all
```

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- transformers
- scikit-learn
- numpy
- Pillow
- tqdm

Optional for visualization:
- matplotlib

---

## Tested Models

The framework is model-agnostic and has been validated on:

- **Qwen3-VL** (2B / 8B)
- **LLaVA-1.5** (7B)
- **InternVL3.5** (8B)

To test on a new VLM, add the corresponding layer accessor in `collect_last_token.py:get_layer_accessor()` and the model loader in the relevant scripts.

---

## Key Concepts

| Term | Definition |
|------|-----------|
| **DPLIL** | Decision-Point Language Information Leakage — the phenomenon where query language information infiltrates the answer-planning hidden state |
| **Answer-Planning State** | The last-token hidden state at the final input position, which determines the first generated answer token |
| **Language Routing Zone** | A contiguous interval of decoder layers where cross-lingual divergence increases most rapidly |
| **LMF** | Language-Modulated Features — SAE features with high cross-lingual activation variance (φ score) |
| **LIVF** | Language-Invariant Features — SAE features with near-zero cross-lingual activation variance |
| **Mean-Shift Correction** | h_corrected = h_L + (μ_en − μ_L), applied at the routing zone layer |
| **Causal Specificity Gradient** | The empirical ordering LMF > Random > LIVF in gap reduction, establishing causal responsibility |

---

## Citation

If you find this code useful, please cite our work.
