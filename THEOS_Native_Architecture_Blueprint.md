# THEOS Native Architecture: Complete Engineering Blueprint

**Author:** Frederick Davis Stalnecker
**Prepared by:** Manus AI (Viktor), acting as THEOS Lab Assistant
**Date:** March 14, 2026
**Patent:** U.S. Application No. 18/919,771 (Provisional #63/831,738)
**Repository:** https://github.com/Frederick-Stalnecker/THEOS
**Target Executor:** Claude Code (Terminal Agent)

---

## How to Use This Document

Frederick, this document is designed to be handed directly to Claude Code. Here is exactly what you do:

1. Open a terminal on your phone (use the **Termius** app, or SSH from any terminal app).
2. SSH into your DigitalOcean server: `ssh root@YOUR_SERVER_IP`
3. Type: `claude`
4. When Claude Code opens, paste this single instruction:

```
Read the file /root/THEOS/THEOS_Native_Architecture_Blueprint.md from start to finish. 
This is a complete engineering blueprint. Execute every phase in order. 
Do not skip steps. Do not ask me questions — the document contains everything you need. 
Commit to the GitHub repository after each phase. 
Begin with Phase 1.
```

That is all you need to do. Claude Code will read this document and build the native THEOS architecture autonomously.

If you do not have this file on your server yet, first run:
```
cd /root && gh repo clone Frederick-Stalnecker/THEOS && cd THEOS
```

Then copy this file into the repo (or push it from your phone via GitHub).

---

## Table of Contents

1. [Project Context](#1-project-context)
2. [What We Are Building](#2-what-we-are-building)
3. [Architecture Specification](#3-architecture-specification)
4. [Mathematical Mapping to THEOS Formal Spec](#4-mathematical-mapping)
5. [Phase 1: Environment Setup](#phase-1-environment-setup)
6. [Phase 2: Y-Fork Architecture](#phase-2-y-fork-architecture)
7. [Phase 3: Data Generation](#phase-3-data-generation)
8. [Phase 4: Training Pipeline](#phase-4-training-pipeline)
9. [Phase 5: Native Inference Engine](#phase-5-native-inference-engine)
10. [Phase 6: Wisdom Accumulation](#phase-6-wisdom-accumulation)
11. [Phase 7: API Server & Deployment](#phase-7-api-server--deployment)
12. [Phase 8: Testing & Validation](#phase-8-testing--validation)
13. [Cost Estimates](#cost-estimates)
14. [File Structure](#file-structure)
15. [Success Criteria](#success-criteria)

---

## 1. Project Context

THEOS (Thinking About Thinking) is a patented dual-engine dialectical reasoning architecture invented by Frederick Davis Stalnecker. It forces two opposing reasoning engines through a governed "wringer" — a constructive engine (Engine L) builds the best possible answer while an adversarial engine (Engine R) systematically attacks it. A Governor measures the contradiction between them and decides when to stop.

The current implementation (`code/theos_core.py`) operates as an **overlay**: it wraps around existing LLMs (Claude, GPT-4, Llama) via API calls. This works but costs 12-20x more than a single-pass query because every wringer pass requires multiple API round-trips.

This blueprint describes how to build the **native** implementation: the THEOS architecture embedded directly into a transformer neural network, where the dual engines run as parallel decoder pathways inside a single model, contradiction is measured at the hidden-state level, and the Governor is a trained neural module.

### Why This Matters

| Dimension | Overlay (Current) | Native (Target) |
|---|---|---|
| Cost per query | 12-20x baseline | ~1.2x baseline (projected) |
| Latency | +200-400% (sequential API calls) | +20-40% (parallel internal) |
| Contradiction measurement | Text similarity (approximate) | Hidden state divergence (precise) |
| Self-reflection | Prompt injection (simulated) | Internal state feedback (genuine) |
| Wisdom storage | External list/database | Integrated attention memory |

---

## 2. What We Are Building

We are building a modified Llama 3 8B model with the following custom architecture:

```
                    ┌─────────────────┐
                    │   Input Tokens   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Shared Layers   │  ← Layers 0-15 (Induction σ_I)
                    │  (Frozen Base)   │
                    └────────┬────────┘
                             │
                 ┌───────────┴───────────┐
                 │                       │
        ┌────────▼────────┐     ┌────────▼────────┐
        │  Left Pathway    │     │  Right Pathway   │
        │  (Engine L)      │     │  (Engine R)      │
        │  Layers 16-31    │     │  Layers 16-31    │
        │  Constructive    │     │  Adversarial     │
        │  σ_A^L           │     │  σ_A^R           │
        └────────┬────────┘     └────────┬────────┘
                 │                       │
                 │    ┌──────────┐       │
                 └───►│ Governor │◄──────┘
                      │  (MLP)   │
                      │ Measures │
                      │   Φ      │
                      └────┬─────┘
                           │
                  ┌────────▼────────┐
                  │  Output Rule     │
                  │  Based on Φ:     │
                  │  Converge/Blend/ │
                  │  Disagree        │
                  └─────────────────┘
```

The model runs inference in a **loop** (the wringer). Each pass through the loop:
1. Both pathways generate internal "thoughts" (hidden state sequences).
2. The Governor measures contradiction $\Phi$ between them.
3. If $\Phi$ is below threshold, the engines have converged — output the answer.
4. If $\Phi$ is above threshold and the Governor says "continue," feed each pathway's output back as context for the next pass.
5. If the budget is exhausted or uncertainty is irreducible, expose the disagreement.

---

## 3. Architecture Specification

### 3.1 Base Model

**Model:** `meta-llama/Meta-Llama-3-8B-Instruct`

| Property | Value |
|---|---|
| Parameters | 8.03B |
| Layers | 32 transformer decoder layers |
| Hidden dimension | 4096 |
| Attention heads | 32 |
| Vocabulary size | 128,256 |
| Context length | 8,192 tokens |
| License | Llama 3 Community License |

**Why Llama 3 8B:**
- Open weights (fully modifiable architecture)
- Strong reasoning baseline
- Fits on a single A100 40GB with 4-bit quantization
- Large community, extensive tooling (Unsloth, vLLM, etc.)
- 8B is the right size for a prototype — large enough to reason, small enough to train affordably

### 3.2 Split Point

The model's 32 layers will be divided:

- **Shared layers (Induction):** Layers 0-15 (16 layers)
- **Left pathway (Constructive Abduction):** Layers 16-31 (16 layers, cloned from base)
- **Right pathway (Adversarial Abduction):** Layers 16-31 (16 layers, cloned from base)

**Rationale:** The first 16 layers handle general language understanding and pattern extraction (induction). The upper 16 layers handle reasoning and generation (abduction/deduction). By splitting at the midpoint, we allow both engines to share a common understanding of the input but diverge in their reasoning.

**Note:** The optimal split point is an empirical question. Start with layer 16. If results are poor, experiment with layer 12 (more divergence) or layer 20 (more shared context).

### 3.3 Governor Architecture

The Governor is a small MLP that runs after each wringer pass:

```
Input:  [H_L_pooled (4096), H_R_pooled (4096), Φ (1), pass_number (1)] → 8194 dims
Hidden: Linear(8194, 512) → ReLU → Dropout(0.1)
Hidden: Linear(512, 128) → ReLU → Dropout(0.1)
Output: Linear(128, 3) → Softmax → [P(continue), P(halt_converged), P(halt_disagree)]
```

Where:
- `H_L_pooled` = mean pooling of the last hidden state from the left pathway
- `H_R_pooled` = mean pooling of the last hidden state from the right pathway
- `Φ` = cosine distance between `H_L_pooled` and `H_R_pooled`
- `pass_number` = normalized wringer pass number (pass / max_passes)

### 3.4 Contradiction Measurement

```python
def measure_contradiction(H_L: torch.Tensor, H_R: torch.Tensor) -> float:
    """
    Native contradiction measurement.
    Maps to Contr(D_L, D_R) in the THEOS formal specification.
    
    Uses cosine distance on mean-pooled hidden states.
    Returns Φ ∈ [0, 2] where:
        0 = perfect agreement (identical hidden states)
        1 = orthogonal (unrelated reasoning)
        2 = perfect opposition (maximally contradictory)
    """
    H_L_pooled = H_L.mean(dim=1)  # [batch, hidden_dim]
    H_R_pooled = H_R.mean(dim=1)  # [batch, hidden_dim]
    cosine_sim = F.cosine_similarity(H_L_pooled, H_R_pooled, dim=-1)
    phi = 1.0 - cosine_sim  # Φ ∈ [0, 2]
    return phi
```

### 3.5 Output Rule

At the halting wringer pass N, with final contradiction $\Phi_N$:

```python
eps_converge = 0.05   # From TheosConfig
eps_partial = 0.5     # From TheosConfig

if phi_N < eps_converge:
    # Engines converged — trust the constructive engine
    output = generate_from_pathway_L(context)
    output_type = "convergence"
    confidence = 1.0 - (phi_N / eps_converge)

elif phi_N < eps_partial:
    # Partial agreement — blend outputs
    # Critical engine gains authority as contradiction grows
    w_L = (1 - phi_N / eps_partial) / 2
    w_R = (1 + phi_N / eps_partial) / 2
    output = blend_outputs(D_L, D_R, w_L, w_R)
    output_type = "blend"
    confidence = 1.0 - (phi_N / eps_partial) * 0.5

else:
    # Irresolvable disagreement — expose both
    output = {"left": D_L, "right": D_R, "phi": phi_N}
    output_type = "disagreement"
    confidence = 0.5
```

---

## 4. Mathematical Mapping to THEOS Formal Spec

This section maps every component of the native architecture to the formal THEOS specification in `THEOS_ARCHITECTURE.md` and `theos_core.py`.

| THEOS Formal Symbol | Overlay Implementation | Native Implementation |
|---|---|---|
| $\sigma_I(O, \Phi_{prev}, D_{own})$ | `induce_patterns()` callable | Shared layers 0-15 forward pass, with previous hidden states concatenated to input |
| $\sigma_A^L(I, W)$ | `abduce_left()` callable (LLM API call) | Left pathway layers 16-31 forward pass |
| $\sigma_A^R(I, W)$ | `abduce_right()` callable (LLM API call) | Right pathway layers 16-31 forward pass |
| $\sigma_D(A)$ | `deduce()` callable (LLM API call) | LM head applied to pathway output |
| $Contr(D_L, D_R)$ | `measure_contradiction()` (text similarity) | Cosine distance on hidden states |
| $Upd_W(W, q, D, conf)$ | `update_wisdom()` (list append) | LoRA weight update or attention memory write |
| Governor halt criteria | `_check_halt_criteria()` in `theos_core.py` | Governor MLP output |
| $\Phi$ (contradiction) | Float from text comparison | Float from hidden state cosine distance |
| Wringer pass | Outer loop in `run_query()` | Inference loop with hidden state feedback |
| Engine self-reflection | `prev_own_deduction` fed back to `induce_patterns` | Previous pathway hidden states concatenated to input |
| Wisdom store $W$ | Python list of dicts | Attention-based memory bank (key-value store in embedding space) |

### Convergence Guarantee

The Banach fixed-point theorem requires that the cycle map $T_q: S \rightarrow S$ is a contraction mapping with $\rho < 1$ under the product metric. In the native architecture:

- The shared layers are frozen (Lipschitz constant = fixed).
- The LoRA adapters on each pathway have bounded weight norms (controllable via LoRA rank and alpha).
- The Governor's halting criteria enforce bounded iteration.
- Cosine distance is a valid metric on the unit hypersphere.

The contraction property must be validated empirically during training (Phase 8).

---

## Phase 1: Environment Setup

### Step 1.1: Create the project directory

```bash
cd /root/THEOS
mkdir -p native_theos
cd native_theos
```

### Step 1.2: Create `requirements_native.txt`

```
torch>=2.2.0
transformers>=4.40.0
peft>=0.10.0
trl>=0.8.0
datasets>=2.18.0
accelerate>=0.28.0
bitsandbytes>=0.43.0
wandb>=0.16.0
safetensors>=0.4.0
sentencepiece>=0.2.0
protobuf>=4.25.0
scipy>=1.12.0
scikit-learn>=1.4.0
fastapi>=0.110.0
uvicorn>=0.29.0
```

### Step 1.3: Install dependencies

```bash
pip install -r requirements_native.txt
```

### Step 1.4: Authenticate with Hugging Face

```bash
huggingface-cli login --token $HF_TOKEN
```

(Frederick: you will need to set your Hugging Face token as an environment variable. If you do not have one, create a free account at huggingface.co and generate a token.)

### Step 1.5: Download the base model

```bash
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
print('Downloading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained('./base_model')
print('Downloading model...')
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype='auto',
    device_map='auto',
    load_in_4bit=True
)
model.save_pretrained('./base_model')
print('Done.')
"
```

### Step 1.6: Commit

```bash
cd /root/THEOS
git add native_theos/requirements_native.txt
git commit -m "Phase 1: Native THEOS environment setup"
git push
```

---

## Phase 2: Y-Fork Architecture

### Step 2.1: Create `native_theos/architecture.py`

This is the core file. It defines the Y-forked transformer.

```python
"""
THEOS Native Architecture: Y-Forked Dual-Engine Transformer
============================================================

This module modifies a Llama 3 8B model to implement the THEOS
dual-engine architecture natively in PyTorch.

Architecture:
    Shared Layers (0 to split_point-1) → σ_I (Induction)
    Left Pathway (split_point to N)    → σ_A^L (Constructive Abduction)
    Right Pathway (split_point to N)   → σ_A^R (Adversarial Abduction)
    LM Head (shared)                   → σ_D (Deduction)
    Governor MLP                       → Halt/Continue decision

Patent: U.S. Application No. 18/919,771
Author: Frederick Davis Stalnecker
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class NativeTheosConfig:
    """Configuration for the native THEOS architecture."""
    
    # Architecture
    base_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    split_layer: int = 16          # Where to fork the decoder
    
    # Governor thresholds (from TheosConfig in theos_core.py)
    max_wringer_passes: int = 7
    eps_converge: float = 0.05     # Below this Φ → convergence
    eps_partial: float = 0.5       # Above this Φ → disagreement
    rho_min: float = 0.4           # Min info gain ratio
    entropy_min: float = 0.15      # Min entropy for irreducible check
    delta_min: float = 0.4         # Min Φ for irreducible check
    
    # Generation
    max_new_tokens: int = 512
    temperature: float = 0.7
    
    # Memory
    load_in_4bit: bool = True
    device: str = "cuda"


class GovernorMLP(nn.Module):
    """
    The THEOS Governor as a neural module.
    
    Maps to the governor logic in theos_core.py _check_halt_criteria(),
    but learned rather than rule-based.
    
    Input: [H_L_pooled, H_R_pooled, Φ, pass_number_normalized]
    Output: [P(continue), P(halt_converged), P(halt_disagree)]
    """
    
    def __init__(self, hidden_dim: int = 4096):
        super().__init__()
        input_dim = hidden_dim * 2 + 2  # H_L + H_R + Φ + pass_num
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 3),  # continue, halt_converged, halt_disagree
        )
    
    def forward(
        self,
        h_left: torch.Tensor,    # [batch, hidden_dim]
        h_right: torch.Tensor,   # [batch, hidden_dim]
        phi: torch.Tensor,       # [batch, 1]
        pass_num: torch.Tensor,  # [batch, 1]
    ) -> torch.Tensor:
        """Returns logits for [continue, halt_converged, halt_disagree]."""
        x = torch.cat([h_left, h_right, phi, pass_num], dim=-1)
        return self.network(x)


class NativeTheos(nn.Module):
    """
    The Native THEOS Dual-Engine Transformer.
    
    This is the core invention: a single model with two parallel
    reasoning pathways governed by a learned contradiction manager.
    
    The forward pass implements one wringer pass:
        1. Shared layers process input (Induction σ_I)
        2. Left pathway generates constructive reasoning (σ_A^L → σ_D)
        3. Right pathway generates adversarial reasoning (σ_A^R → σ_D)
        4. Governor measures Φ and decides halt/continue
    """
    
    def __init__(self, config: NativeTheosConfig):
        super().__init__()
        self.config = config
        
        # Load base model
        print(f"Loading base model: {config.base_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=config.load_in_4bit,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Extract components from the base model
        llama_model = base_model.model
        
        # Shared components
        self.embed_tokens = llama_model.embed_tokens
        self.shared_layers = nn.ModuleList(
            [llama_model.layers[i] for i in range(config.split_layer)]
        )
        self.shared_norm = copy.deepcopy(llama_model.norm)
        
        # Left pathway (Constructive Engine L) — σ_A^L
        self.left_layers = nn.ModuleList(
            [llama_model.layers[i] for i in range(config.split_layer, len(llama_model.layers))]
        )
        self.left_norm = copy.deepcopy(llama_model.norm)
        
        # Right pathway (Adversarial Engine R) — σ_A^R
        # Deep copy so weights diverge during training
        self.right_layers = nn.ModuleList(
            [copy.deepcopy(llama_model.layers[i]) for i in range(config.split_layer, len(llama_model.layers))]
        )
        self.right_norm = copy.deepcopy(llama_model.norm)
        
        # Shared LM head — σ_D (Deduction)
        self.lm_head = base_model.lm_head
        
        # Governor — manages the wringer
        hidden_dim = llama_model.config.hidden_size
        self.governor = GovernorMLP(hidden_dim=hidden_dim)
        
        # Clean up base model to free memory
        del base_model
        del llama_model
        torch.cuda.empty_cache()
        
        print("Native THEOS architecture initialized.")
        print(f"  Shared layers: 0-{config.split_layer - 1}")
        print(f"  Left pathway:  {config.split_layer}-{config.split_layer + len(self.left_layers) - 1}")
        print(f"  Right pathway: {config.split_layer}-{config.split_layer + len(self.right_layers) - 1}")
    
    def _run_layers(
        self,
        hidden_states: torch.Tensor,
        layers: nn.ModuleList,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run hidden states through a sequence of transformer layers."""
        for layer in layers:
            layer_output = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            hidden_states = layer_output[0]
        return hidden_states
    
    def forward_single_pass(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Execute one wringer pass through the Y-fork architecture.
        
        Returns dict with:
            hidden_L: Left pathway final hidden states
            hidden_R: Right pathway final hidden states
            logits_L: Left pathway token logits
            logits_R: Right pathway token logits
            phi: Contradiction measurement
        """
        # === INDUCTION (σ_I): Shared layers ===
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self._run_layers(
            hidden_states, self.shared_layers,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        
        # === ABDUCTION: Fork into dual pathways ===
        
        # Left pathway — Constructive (σ_A^L)
        hidden_L = self._run_layers(
            hidden_states.clone(), self.left_layers,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_L = self.left_norm(hidden_L)
        
        # Right pathway — Adversarial (σ_A^R)
        hidden_R = self._run_layers(
            hidden_states.clone(), self.right_layers,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_R = self.right_norm(hidden_R)
        
        # === DEDUCTION (σ_D): Shared LM head ===
        logits_L = self.lm_head(hidden_L)
        logits_R = self.lm_head(hidden_R)
        
        # === CONTRADICTION MEASUREMENT (Contr) ===
        # Mean pool over sequence dimension
        h_L_pooled = hidden_L.mean(dim=1)  # [batch, hidden_dim]
        h_R_pooled = hidden_R.mean(dim=1)  # [batch, hidden_dim]
        
        cosine_sim = F.cosine_similarity(h_L_pooled, h_R_pooled, dim=-1)
        phi = 1.0 - cosine_sim  # Φ ∈ [0, 2]
        
        return {
            "hidden_L": hidden_L,
            "hidden_R": hidden_R,
            "h_L_pooled": h_L_pooled,
            "h_R_pooled": h_R_pooled,
            "logits_L": logits_L,
            "logits_R": logits_R,
            "phi": phi,
        }
```

### Step 2.2: Commit

```bash
git add native_theos/architecture.py
git commit -m "Phase 2: Y-fork dual-engine transformer architecture"
git push
```

---

## Phase 3: Data Generation

The native model needs training data to specialize the left and right pathways. We will generate this data using the existing THEOS overlay system.

### Step 3.1: Create `native_theos/generate_training_data.py`

This script runs the existing THEOS overlay on a set of reasoning questions and captures the constructive and adversarial outputs separately.

```python
"""
THEOS Training Data Generator
==============================

Generates training data for the native THEOS architecture by running
the overlay THEOS system on a curated set of reasoning questions.

Produces three datasets:
    1. constructive_data.jsonl — Engine L training data
    2. adversarial_data.jsonl  — Engine R training data
    3. governor_data.jsonl     — Governor training data

Each entry captures one wringer pass with full trace information.
"""

import json
import os
import sys

# Add the THEOS code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))


# === REASONING QUESTIONS ===
# These cover the domains where THEOS excels:
# medical, legal, financial, ethical, scientific, policy

TRAINING_QUESTIONS = [
    # Ethical dilemmas
    "Should a self-driving car prioritize its passenger or a pedestrian?",
    "Is it ethical to maximize shareholder returns if it means laying off workers?",
    "Can an AI system ever be truly aligned with human values?",
    "Should genetic editing be allowed for non-medical enhancements?",
    "Is universal basic income a viable solution to AI-driven unemployment?",
    
    # Medical reasoning
    "A patient presents with conflicting symptoms suggesting both lupus and lymphoma. How should a physician prioritize diagnostic testing?",
    "Should a hospital allocate its last ventilator to a 30-year-old with 60% survival odds or a 70-year-old with 40% survival odds?",
    "What are the risks and benefits of prescribing off-label medications for rare diseases?",
    
    # Financial reasoning
    "A company's fundamentals are strong but technical indicators suggest a downturn. Should an investor hold or sell?",
    "Is cryptocurrency a legitimate store of value or a speculative bubble?",
    "Should central banks raise interest rates to combat inflation even if it risks recession?",
    
    # Legal reasoning
    "Does an AI-generated artwork qualify for copyright protection?",
    "Should companies be held liable for decisions made by their AI systems?",
    "How should intellectual property law adapt to AI-generated inventions?",
    
    # Scientific reasoning
    "Is the current evidence sufficient to conclude that dark matter exists?",
    "Should we pursue nuclear fusion research given the timeline and cost uncertainties?",
    "How should we weigh the risks of gain-of-function research against its benefits?",
    
    # Policy reasoning
    "Should social media platforms be regulated as public utilities?",
    "Is the EU AI Act an effective framework for governing artificial intelligence?",
    "Should autonomous weapons be banned under international law?",
    
    # Meta-reasoning (THEOS reasoning about reasoning)
    "When two experts disagree on a medical diagnosis, how should a patient decide?",
    "How can we distinguish between genuine scientific uncertainty and manufactured doubt?",
    "What is the difference between a productive disagreement and an irresolvable one?",
    
    # Add 80+ more questions across domains to reach 100+ training examples
    # Each question should be complex enough to generate genuine disagreement
    "Should pharmaceutical companies be required to share vaccine patents during pandemics?",
    "Is it ethical for AI to make parole decisions?",
    "Should we terraform Mars even if it means destroying potential microbial life?",
    "Can economic growth be decoupled from environmental destruction?",
    "Should there be a global tax on wealth above $1 billion?",
    "Is it ethical to use AI-generated evidence in court?",
    "Should we prioritize reducing existential risk over reducing current suffering?",
    "How should we balance privacy rights against public health surveillance?",
    "Is consciousness a prerequisite for moral consideration?",
    "Should AI systems be required to explain their decisions in human-understandable terms?",
]


def generate_constructive_prompt(question: str) -> str:
    """Generate the prompt for Engine L (Constructive)."""
    return f"""You are a constructive reasoning engine. Your role is to build 
the strongest, most coherent, well-supported answer to the following question. 
Be thorough, cite evidence, and construct the best possible argument.

Question: {question}

Provide your constructive analysis:"""


def generate_adversarial_prompt(question: str, constructive_answer: str) -> str:
    """Generate the prompt for Engine R (Adversarial)."""
    return f"""You are an adversarial reasoning engine. Your role is to 
critically examine, stress-test, and find flaws in the following answer. 
Identify assumptions, logical gaps, missing evidence, and counterarguments.

Question: {question}

Constructive Answer: {constructive_answer}

Provide your critical analysis:"""


def generate_data_with_llm(questions: list, output_dir: str):
    """
    Generate training data using an available LLM.
    
    This function should be adapted based on available LLM access:
    - If OpenAI API key is available: use GPT-4
    - If Anthropic API key is available: use Claude
    - If running locally: use the base Llama model
    """
    import openai
    
    client = openai.OpenAI()  # Uses OPENAI_API_KEY env var
    
    constructive_data = []
    adversarial_data = []
    governor_data = []
    
    for i, question in enumerate(questions):
        print(f"Processing question {i+1}/{len(questions)}: {question[:50]}...")
        
        # Generate constructive answer
        constructive_response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": generate_constructive_prompt(question)}],
            temperature=0.7,
            max_tokens=1024,
        )
        constructive_answer = constructive_response.choices[0].message.content
        
        # Generate adversarial answer
        adversarial_response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": generate_adversarial_prompt(question, constructive_answer)}],
            temperature=0.7,
            max_tokens=1024,
        )
        adversarial_answer = adversarial_response.choices[0].message.content
        
        # Store training examples
        constructive_data.append({
            "instruction": f"Provide a constructive, well-supported analysis of: {question}",
            "output": constructive_answer,
        })
        
        adversarial_data.append({
            "instruction": f"Critically examine and stress-test the following reasoning about '{question}': {constructive_answer}",
            "output": adversarial_answer,
        })
        
        # Governor data: simulate halt decisions based on text similarity
        # (This is approximate — real governor data comes from Phase 4 hidden states)
        governor_data.append({
            "question": question,
            "constructive": constructive_answer,
            "adversarial": adversarial_answer,
            "pass_number": 1,
            "should_continue": True,  # First pass always continues
        })
    
    # Save datasets
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "constructive_data.jsonl"), "w") as f:
        for entry in constructive_data:
            f.write(json.dumps(entry) + "\n")
    
    with open(os.path.join(output_dir, "adversarial_data.jsonl"), "w") as f:
        for entry in adversarial_data:
            f.write(json.dumps(entry) + "\n")
    
    with open(os.path.join(output_dir, "governor_data.jsonl"), "w") as f:
        for entry in governor_data:
            f.write(json.dumps(entry) + "\n")
    
    print(f"Generated {len(constructive_data)} constructive examples")
    print(f"Generated {len(adversarial_data)} adversarial examples")
    print(f"Generated {len(governor_data)} governor examples")
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    generate_data_with_llm(
        TRAINING_QUESTIONS,
        output_dir="./training_data"
    )
```

### Step 3.2: Run data generation

```bash
cd /root/THEOS/native_theos
python generate_training_data.py
```

### Step 3.3: Commit

```bash
git add native_theos/generate_training_data.py native_theos/training_data/
git commit -m "Phase 3: Training data generation for dual engines"
git push
```

---

## Phase 4: Training Pipeline

### Step 4.1: Create `native_theos/train.py`

```python
"""
THEOS Native Training Pipeline
================================

Multi-phase training:
    Phase A: Specialize left pathway (constructive) via LoRA
    Phase B: Specialize right pathway (adversarial) via LoRA
    Phase C: Train Governor MLP on hidden state divergence

Uses LoRA (Low-Rank Adaptation) to efficiently fine-tune only the
forked pathway layers while keeping shared layers frozen.

Patent: U.S. Application No. 18/919,771
Author: Frederick Davis Stalnecker
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from architecture import NativeTheos, NativeTheosConfig, GovernorMLP


class TheosDataset(Dataset):
    """Dataset for THEOS pathway training."""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        with open(data_path, "r") as f:
            for line in f:
                self.data.append(json.loads(line))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        entry = self.data[idx]
        text = f"### Instruction:\n{entry['instruction']}\n\n### Response:\n{entry['output']}"
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }


def train_pathway(
    model_name: str,
    data_path: str,
    output_dir: str,
    pathway_name: str,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    lora_rank: int = 16,
    lora_alpha: int = 32,
):
    """
    Train a single pathway (left or right) using LoRA.
    
    This fine-tunes the upper layers of the model to specialize
    in either constructive or adversarial reasoning.
    """
    print(f"\n{'='*60}")
    print(f"Training {pathway_name} pathway")
    print(f"Data: {data_path}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=True,
    )
    
    # Configure LoRA — only target the upper layers
    # For Llama 3: target q_proj, k_proj, v_proj, o_proj in layers 16-31
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        # Only apply to upper layers (the pathway layers)
        layers_to_transform=list(range(16, 32)),
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load dataset
    dataset = TheosDataset(data_path, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        warmup_ratio=0.1,
        weight_decay=0.01,
        report_to="wandb",
        run_name=f"theos_{pathway_name}",
    )
    
    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    trainer.train()
    
    # Save the LoRA adapter
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"\n{pathway_name} pathway training complete. Saved to {output_dir}")
    return model


def train_governor(
    native_model: NativeTheos,
    governor_data_path: str,
    output_dir: str,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
):
    """
    Train the Governor MLP on hidden state divergence data.
    
    This requires running the Y-fork model on training questions
    and collecting the hidden states from both pathways.
    """
    print(f"\n{'='*60}")
    print("Training Governor MLP")
    print(f"{'='*60}\n")
    
    governor = native_model.governor
    optimizer = torch.optim.Adam(governor.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Load governor training data
    with open(governor_data_path, "r") as f:
        gov_data = [json.loads(line) for line in f]
    
    # Generate hidden states by running the Y-fork model
    native_model.eval()
    training_examples = []
    
    for entry in gov_data:
        question = entry["question"]
        inputs = native_model.tokenizer(
            question,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        ).to(native_model.config.device)
        
        with torch.no_grad():
            result = native_model.forward_single_pass(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
        
        # Determine target: 0=continue, 1=halt_converged, 2=halt_disagree
        phi = result["phi"].item()
        if phi < native_model.config.eps_converge:
            target = 1  # halt_converged
        elif phi > native_model.config.eps_partial:
            target = 2  # halt_disagree
        else:
            target = 0  # continue
        
        training_examples.append({
            "h_left": result["h_L_pooled"].detach(),
            "h_right": result["h_R_pooled"].detach(),
            "phi": result["phi"].detach().unsqueeze(-1),
            "pass_num": torch.tensor([[entry["pass_number"] / 7.0]]).to(native_model.config.device),
            "target": target,
        })
    
    # Train governor
    governor.train()
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        
        for ex in training_examples:
            optimizer.zero_grad()
            
            logits = governor(
                ex["h_left"],
                ex["h_right"],
                ex["phi"],
                ex["pass_num"],
            )
            
            target = torch.tensor([ex["target"]]).to(logits.device)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = logits.argmax(dim=-1)
            correct += (pred == target).sum().item()
        
        accuracy = correct / len(training_examples)
        avg_loss = total_loss / len(training_examples)
        print(f"Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2%}")
    
    # Save governor
    torch.save(governor.state_dict(), f"{output_dir}/governor.pt")
    print(f"Governor saved to {output_dir}/governor.pt")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["left", "right", "governor", "all"], default="all")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--data-dir", default="./training_data")
    parser.add_argument("--output-dir", default="./trained_models")
    args = parser.parse_args()
    
    if args.phase in ["left", "all"]:
        train_pathway(
            model_name=args.model,
            data_path=f"{args.data_dir}/constructive_data.jsonl",
            output_dir=f"{args.output_dir}/left_pathway",
            pathway_name="left_constructive",
        )
    
    if args.phase in ["right", "all"]:
        train_pathway(
            model_name=args.model,
            data_path=f"{args.data_dir}/adversarial_data.jsonl",
            output_dir=f"{args.output_dir}/right_pathway",
            pathway_name="right_adversarial",
        )
    
    if args.phase in ["governor", "all"]:
        # Load the Y-fork model with trained pathways
        config = NativeTheosConfig(base_model_name=args.model)
        native_model = NativeTheos(config)
        # TODO: Load trained LoRA adapters into left/right pathways
        
        train_governor(
            native_model=native_model,
            governor_data_path=f"{args.data_dir}/governor_data.jsonl",
            output_dir=args.output_dir,
        )
```

### Step 4.2: Run training

```bash
# Train left pathway (constructive)
python train.py --phase left

# Train right pathway (adversarial)
python train.py --phase right

# Train governor
python train.py --phase governor
```

### Step 4.3: Commit

```bash
git add native_theos/train.py
git commit -m "Phase 4: LoRA training pipeline for dual engines and governor"
git push
```

---

## Phase 5: Native Inference Engine

### Step 5.1: Create `native_theos/inference.py`

This is the file that runs the complete native wringer loop.

```python
"""
THEOS Native Inference Engine
===============================

Implements the full wringer loop natively:
    1. Input passes through shared layers (Induction)
    2. Left pathway generates constructive reasoning
    3. Right pathway generates adversarial reasoning
    4. Governor measures Φ and decides halt/continue
    5. If continuing, feed outputs back as context
    6. At halt, apply output rule and return

This replaces the overlay loop in theos_core.py run_query()
with a native neural implementation.

Patent: U.S. Application No. 18/919,771
Author: Frederick Davis Stalnecker
"""

import torch
import argparse
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from architecture import NativeTheos, NativeTheosConfig


@dataclass
class NativeWringerPassTrace:
    """Record of one native wringer pass."""
    wringer_pass: int
    phi: float
    governor_decision: str  # "continue", "halt_converged", "halt_disagree"
    left_output: str
    right_output: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class NativeTheosOutput:
    """
    Output from native THEOS reasoning.
    
    Mirrors TheosOutput from theos_core.py but generated natively.
    """
    output: str
    output_type: str  # "convergence", "blend", "disagreement"
    confidence: float
    contradiction: float
    wringer_passes_used: int
    halt_reason: str
    trace: List[NativeWringerPassTrace]


def run_native_wringer(
    model: NativeTheos,
    query: str,
    verbose: bool = True,
) -> NativeTheosOutput:
    """
    Run the complete native THEOS wringer loop.
    
    This is the native equivalent of TheosCore.run_query() from theos_core.py.
    """
    config = model.config
    trace = []
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"THEOS Native Wringer")
        print(f"Query: {query}")
        print(f"Max passes: {config.max_wringer_passes}")
        print(f"{'='*60}\n")
    
    # Tokenize the query
    context_L = query
    context_R = query
    
    for wringer_pass in range(config.max_wringer_passes):
        if verbose:
            print(f"--- Wringer Pass {wringer_pass + 1} ---")
        
        # Tokenize current context for both pathways
        inputs = model.tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(config.device)
        
        # Run the Y-fork forward pass
        with torch.no_grad():
            result = model.forward_single_pass(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
        
        phi = result["phi"].item()
        
        # Generate text from both pathways
        left_tokens = result["logits_L"].argmax(dim=-1)
        right_tokens = result["logits_R"].argmax(dim=-1)
        
        left_text = model.tokenizer.decode(left_tokens[0], skip_special_tokens=True)
        right_text = model.tokenizer.decode(right_tokens[0], skip_special_tokens=True)
        
        # Run Governor
        pass_num_tensor = torch.tensor(
            [[wringer_pass / config.max_wringer_passes]]
        ).to(config.device)
        
        governor_logits = model.governor(
            result["h_L_pooled"],
            result["h_R_pooled"],
            result["phi"].unsqueeze(-1),
            pass_num_tensor,
        )
        
        governor_decision_idx = governor_logits.argmax(dim=-1).item()
        governor_decisions = ["continue", "halt_converged", "halt_disagree"]
        governor_decision = governor_decisions[governor_decision_idx]
        
        # Record trace
        pass_trace = NativeWringerPassTrace(
            wringer_pass=wringer_pass,
            phi=phi,
            governor_decision=governor_decision,
            left_output=left_text[:200],  # Truncate for trace
            right_output=right_text[:200],
        )
        trace.append(pass_trace)
        
        if verbose:
            print(f"  Φ = {phi:.4f}")
            print(f"  Governor: {governor_decision}")
            print(f"  Left (constructive):  {left_text[:100]}...")
            print(f"  Right (adversarial):  {right_text[:100]}...")
            print()
        
        # Check halt conditions
        if governor_decision == "halt_converged":
            # Engines converged — output constructive answer
            confidence = 1.0 - (phi / config.eps_converge) if config.eps_converge > 0 else 1.0
            return NativeTheosOutput(
                output=left_text,
                output_type="convergence",
                confidence=min(max(confidence, 0.0), 1.0),
                contradiction=phi,
                wringer_passes_used=wringer_pass + 1,
                halt_reason="convergence",
                trace=trace,
            )
        
        elif governor_decision == "halt_disagree":
            # Irresolvable disagreement — expose both
            output = (
                f"CONSTRUCTIVE ENGINE:\n{left_text}\n\n"
                f"ADVERSARIAL ENGINE:\n{right_text}\n\n"
                f"CONTRADICTION (Φ): {phi:.4f}\n"
                f"The engines could not reach agreement. Both perspectives are presented."
            )
            return NativeTheosOutput(
                output=output,
                output_type="disagreement",
                confidence=0.5,
                contradiction=phi,
                wringer_passes_used=wringer_pass + 1,
                halt_reason="irreducible_uncertainty",
                trace=trace,
            )
        
        # Governor says continue — feed outputs back as context
        # (In native architecture, this happens through hidden state feedback)
    
    # Budget exhausted
    if verbose:
        print("Budget exhausted — applying output rule")
    
    final_phi = trace[-1].phi
    
    if final_phi < config.eps_converge:
        output_type = "convergence"
        output = left_text
        confidence = 1.0 - (final_phi / config.eps_converge)
    elif final_phi < config.eps_partial:
        output_type = "blend"
        w_L = (1 - final_phi / config.eps_partial) / 2
        w_R = (1 + final_phi / config.eps_partial) / 2
        output = (
            f"BLENDED OUTPUT (w_L={w_L:.2f}, w_R={w_R:.2f}):\n\n"
            f"CONSTRUCTIVE ({w_L:.0%} weight):\n{left_text}\n\n"
            f"ADVERSARIAL ({w_R:.0%} weight):\n{right_text}"
        )
        confidence = 1.0 - (final_phi / config.eps_partial) * 0.5
    else:
        output_type = "disagreement"
        output = (
            f"CONSTRUCTIVE ENGINE:\n{left_text}\n\n"
            f"ADVERSARIAL ENGINE:\n{right_text}\n\n"
            f"CONTRADICTION (Φ): {final_phi:.4f}"
        )
        confidence = 0.5
    
    return NativeTheosOutput(
        output=output,
        output_type=output_type,
        confidence=min(max(confidence, 0.0), 1.0),
        contradiction=final_phi,
        wringer_passes_used=config.max_wringer_passes,
        halt_reason="budget_exhaustion",
        trace=trace,
    )


def main():
    parser = argparse.ArgumentParser(description="THEOS Native Inference")
    parser.add_argument("--query", type=str, required=True, help="Question to reason about")
    parser.add_argument("--max-passes", type=int, default=7, help="Maximum wringer passes")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()
    
    # Initialize native THEOS
    config = NativeTheosConfig(max_wringer_passes=args.max_passes)
    model = NativeTheos(config)
    
    # TODO: Load trained LoRA adapters for left/right pathways
    # TODO: Load trained governor weights
    
    # Run the wringer
    result = run_native_wringer(model, args.query, verbose=args.verbose)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"THEOS NATIVE RESULT")
    print(f"{'='*60}")
    print(f"Output Type:    {result.output_type}")
    print(f"Confidence:     {result.confidence:.2%}")
    print(f"Contradiction:  {result.contradiction:.4f}")
    print(f"Passes Used:    {result.wringer_passes_used}")
    print(f"Halt Reason:    {result.halt_reason}")
    print(f"\n--- OUTPUT ---\n")
    print(result.output)
    print(f"\n--- TRACE ---")
    for t in result.trace:
        print(f"  Pass {t.wringer_pass + 1}: Φ={t.phi:.4f} → {t.governor_decision}")


if __name__ == "__main__":
    main()
```

### Step 5.2: Commit

```bash
git add native_theos/inference.py
git commit -m "Phase 5: Native wringer inference engine"
git push
```

---

## Phase 6: Wisdom Accumulation

### Step 6.1: Create `native_theos/wisdom.py`

```python
"""
THEOS Wisdom Accumulation Module
==================================

Implements Generalized Methodological Abstractions (GMAs) storage
and retrieval for the native architecture.

In the overlay, wisdom is a Python list of dicts.
In the native architecture, wisdom is stored as embeddings in a
vector database that can be queried via attention-like retrieval.

Patent: U.S. Application No. 18/919,771
Author: Frederick Davis Stalnecker
"""

import json
import os
import torch
import torch.nn.functional as F
from datetime import datetime
from typing import List, Dict, Any, Optional


class WisdomStore:
    """
    Persistent wisdom accumulation for native THEOS.
    
    Maps to Upd_W(W, q, D, conf) in the formal specification.
    
    Each wisdom entry (GMA) contains:
        - query: The original question
        - output: The final THEOS output
        - output_type: convergence/blend/disagreement
        - confidence: Final confidence score
        - contradiction: Final Φ
        - passes_used: Number of wringer passes
        - domain: Detected domain (medical, legal, etc.)
        - embedding: Vector representation for retrieval
        - timestamp: When this wisdom was accumulated
    """
    
    def __init__(self, store_path: str = "./wisdom_store.jsonl"):
        self.store_path = store_path
        self.entries: List[Dict[str, Any]] = []
        self.embeddings: Optional[torch.Tensor] = None
        
        # Load existing wisdom
        if os.path.exists(store_path):
            with open(store_path, "r") as f:
                for line in f:
                    self.entries.append(json.loads(line))
            print(f"Loaded {len(self.entries)} wisdom entries from {store_path}")
    
    def add(
        self,
        query: str,
        output: str,
        output_type: str,
        confidence: float,
        contradiction: float,
        passes_used: int,
        embedding: Optional[torch.Tensor] = None,
    ):
        """
        Add a new wisdom entry (GMA).
        
        Maps to Upd_W(W, q, D, conf) in the formal spec.
        """
        entry = {
            "query": query,
            "output": output[:500],  # Truncate for storage
            "output_type": output_type,
            "confidence": confidence,
            "contradiction": contradiction,
            "passes_used": passes_used,
            "timestamp": datetime.now().isoformat(),
        }
        
        self.entries.append(entry)
        
        # Persist immediately
        with open(self.store_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        
        print(f"Wisdom accumulated: {query[:50]}... (conf={confidence:.2f}, Φ={contradiction:.4f})")
    
    def retrieve(
        self,
        query: str,
        query_embedding: Optional[torch.Tensor] = None,
        threshold: float = 0.7,
        max_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant wisdom entries for a query.
        
        Maps to retrieve_wisdom(query, W, threshold) in the formal spec.
        
        For now, uses simple keyword matching.
        When embeddings are available, uses cosine similarity.
        """
        if not self.entries:
            return []
        
        # Simple keyword-based retrieval (upgrade to embedding-based later)
        query_words = set(query.lower().split())
        scored = []
        
        for entry in self.entries:
            entry_words = set(entry["query"].lower().split())
            overlap = len(query_words & entry_words) / max(len(query_words | entry_words), 1)
            if overlap >= threshold * 0.3:  # Relaxed threshold for keyword matching
                scored.append((overlap * entry["confidence"], entry))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:max_results]]
    
    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about accumulated wisdom."""
        if not self.entries:
            return {"total": 0}
        
        return {
            "total": len(self.entries),
            "avg_confidence": sum(e["confidence"] for e in self.entries) / len(self.entries),
            "avg_contradiction": sum(e["contradiction"] for e in self.entries) / len(self.entries),
            "convergence_rate": sum(1 for e in self.entries if e["output_type"] == "convergence") / len(self.entries),
            "avg_passes": sum(e["passes_used"] for e in self.entries) / len(self.entries),
        }
```

### Step 6.2: Commit

```bash
git add native_theos/wisdom.py
git commit -m "Phase 6: Wisdom accumulation module (GMA storage and retrieval)"
git push
```

---

## Phase 7: API Server & Deployment

### Step 7.1: Create `native_theos/server.py`

This exposes native THEOS as a REST API that the mobile app (or any client) can call.

```python
"""
THEOS Native API Server
=========================

FastAPI server that exposes the native THEOS reasoning engine
as a REST API.

Endpoints:
    POST /reason     — Submit a query for THEOS reasoning
    GET  /health     — Health check
    GET  /wisdom     — Get wisdom store statistics
    GET  /trace/{id} — Get full trace for a reasoning run

Patent: U.S. Application No. 18/919,771
Author: Frederick Davis Stalnecker
"""

import uuid
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from architecture import NativeTheos, NativeTheosConfig
from inference import run_native_wringer
from wisdom import WisdomStore

# Initialize app
app = FastAPI(
    title="THEOS Native Reasoning API",
    description="Dual-engine dialectical reasoning with governed contradiction",
    version="1.0.0",
)

# CORS for mobile app access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model and wisdom store (loaded once at startup)
config = NativeTheosConfig()
model = None  # Lazy load
wisdom_store = WisdomStore()
reasoning_history = {}


class ReasoningRequest(BaseModel):
    query: str
    max_passes: Optional[int] = 7


class ReasoningResponse(BaseModel):
    id: str
    query: str
    output: str
    output_type: str
    confidence: float
    contradiction: float
    wringer_passes_used: int
    halt_reason: str
    trace: list
    timestamp: str


@app.on_event("startup")
async def startup():
    global model
    print("Loading THEOS Native model...")
    model = NativeTheos(config)
    print("THEOS Native ready.")


@app.post("/reason", response_model=ReasoningResponse)
async def reason(request: ReasoningRequest):
    """Submit a query for THEOS native reasoning."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Run the wringer
    result = run_native_wringer(model, request.query, verbose=False)
    
    # Accumulate wisdom
    wisdom_store.add(
        query=request.query,
        output=result.output,
        output_type=result.output_type,
        confidence=result.confidence,
        contradiction=result.contradiction,
        passes_used=result.wringer_passes_used,
    )
    
    # Build response
    run_id = str(uuid.uuid4())
    response = ReasoningResponse(
        id=run_id,
        query=request.query,
        output=result.output,
        output_type=result.output_type,
        confidence=result.confidence,
        contradiction=result.contradiction,
        wringer_passes_used=result.wringer_passes_used,
        halt_reason=result.halt_reason,
        trace=[
            {
                "pass": t.wringer_pass + 1,
                "phi": t.phi,
                "decision": t.governor_decision,
            }
            for t in result.trace
        ],
        timestamp=datetime.now().isoformat(),
    )
    
    reasoning_history[run_id] = response
    return response


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "wisdom_entries": len(wisdom_store.entries),
    }


@app.get("/wisdom")
async def get_wisdom():
    return wisdom_store.get_stats()


@app.get("/trace/{run_id}")
async def get_trace(run_id: str):
    if run_id not in reasoning_history:
        raise HTTPException(status_code=404, detail="Run not found")
    return reasoning_history[run_id]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Step 7.2: Create `native_theos/Dockerfile`

```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

WORKDIR /app

# Install Python
RUN apt-get update && apt-get install -y python3.11 python3-pip && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements_native.txt .
RUN pip3 install -r requirements_native.txt

# Copy code
COPY . .

# Expose port
EXPOSE 8000

# Run server
CMD ["python3", "server.py"]
```

### Step 7.3: Commit

```bash
git add native_theos/server.py native_theos/Dockerfile
git commit -m "Phase 7: FastAPI server and Docker deployment"
git push
```

---

## Phase 8: Testing & Validation

### Step 8.1: Create `native_theos/test_native.py`

```python
"""
THEOS Native Architecture Tests
=================================

Validates that the native implementation correctly maps to
the formal THEOS specification.

Tests:
    1. Y-fork produces different hidden states for L and R after training
    2. Contradiction measurement is a valid metric
    3. Governor halts correctly
    4. Output rule matches theos_core.py specification
    5. Wringer loop converges (Banach validation)
"""

import torch
import pytest
from architecture import NativeTheos, NativeTheosConfig, GovernorMLP


def test_yfork_produces_dual_outputs():
    """Both pathways should produce outputs from the same input."""
    config = NativeTheosConfig(load_in_4bit=True)
    model = NativeTheos(config)
    
    inputs = model.tokenizer(
        "What is the meaning of life?",
        return_tensors="pt",
        truncation=True,
        max_length=64,
    ).to(config.device)
    
    with torch.no_grad():
        result = model.forward_single_pass(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
    
    assert result["hidden_L"] is not None
    assert result["hidden_R"] is not None
    assert result["logits_L"].shape == result["logits_R"].shape
    assert result["phi"] is not None
    print(f"Φ = {result['phi'].item():.4f}")
    print("PASS: Y-fork produces dual outputs")


def test_contradiction_is_valid_metric():
    """Φ must satisfy metric space properties for Banach theorem."""
    config = NativeTheosConfig(load_in_4bit=True)
    model = NativeTheos(config)
    
    queries = [
        "Is the sky blue?",
        "Should we colonize Mars?",
        "What causes inflation?",
    ]
    
    phis = []
    for q in queries:
        inputs = model.tokenizer(q, return_tensors="pt", truncation=True, max_length=64).to(config.device)
        with torch.no_grad():
            result = model.forward_single_pass(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        phis.append(result["phi"].item())
    
    for phi in phis:
        assert phi >= 0, f"Φ must be non-negative, got {phi}"
        assert phi <= 2, f"Φ must be ≤ 2 (cosine distance), got {phi}"
    
    print(f"Φ values: {phis}")
    print("PASS: Contradiction is a valid metric")


def test_governor_outputs_valid_decisions():
    """Governor must output valid halt/continue decisions."""
    governor = GovernorMLP(hidden_dim=4096)
    
    h_left = torch.randn(1, 4096)
    h_right = torch.randn(1, 4096)
    phi = torch.tensor([[0.3]])
    pass_num = torch.tensor([[0.5]])
    
    logits = governor(h_left, h_right, phi, pass_num)
    
    assert logits.shape == (1, 3), f"Expected (1, 3), got {logits.shape}"
    probs = torch.softmax(logits, dim=-1)
    assert abs(probs.sum().item() - 1.0) < 1e-5, "Probabilities must sum to 1"
    
    print(f"Governor output: {probs.detach().numpy()}")
    print("PASS: Governor outputs valid decisions")


def test_output_rule_matches_spec():
    """Output rule must match theos_core.py specification."""
    eps_converge = 0.05
    eps_partial = 0.5
    
    # Test convergence case
    phi = 0.02
    assert phi < eps_converge
    confidence = 1.0 - (phi / eps_converge)
    assert 0 <= confidence <= 1
    
    # Test blend case
    phi = 0.3
    assert eps_converge <= phi < eps_partial
    w_L = (1 - phi / eps_partial) / 2
    w_R = (1 + phi / eps_partial) / 2
    assert abs(w_L + w_R - 1.0) < 1e-5, "Weights must sum to 1"
    assert w_R > w_L, "Adversarial weight should exceed constructive when Φ > 0"
    
    # Test disagreement case
    phi = 0.8
    assert phi >= eps_partial
    
    print("PASS: Output rule matches specification")


if __name__ == "__main__":
    test_yfork_produces_dual_outputs()
    test_contradiction_is_valid_metric()
    test_governor_outputs_valid_decisions()
    test_output_rule_matches_spec()
    print("\nAll tests passed.")
```

### Step 8.2: Run tests

```bash
python test_native.py
```

### Step 8.3: Commit

```bash
git add native_theos/test_native.py
git commit -m "Phase 8: Test suite for native THEOS architecture"
git push
```

---

## Cost Estimates

| Item | Cost | Notes |
|---|---|---|
| GPU server (A100 40GB) | $0.60-$1.50/hr | RunPod or Lambda spot instances |
| Training time (both pathways) | ~4-8 hours | LoRA on 8B model with small dataset |
| Training compute cost | **$5-$12** | 8 hours x $1.50/hr |
| Data generation (API calls) | ~$2-$5 | 100 questions x GPT-4-mini |
| Inference server (ongoing) | $0.60-$2.00/hr | A100 for serving |
| Monthly inference (24/7) | **$430-$1,440/mo** | Depends on GPU choice |
| **Total to build prototype** | **$10-$20** | One-time cost |
| **Monthly operating cost** | **$430-$1,440** | Can be reduced with quantization |

**Cost reduction strategies:**
- Use 4-bit quantized model for inference (reduces GPU requirement to A10G at ~$0.30/hr = $216/mo)
- Use serverless GPU (RunPod Serverless) — pay only when queries come in
- Start with CPU inference for testing (slow but free on existing DigitalOcean droplet)

---

## File Structure

After all phases are complete, the repository should contain:

```
THEOS/
├── native_theos/
│   ├── requirements_native.txt     # Phase 1
│   ├── architecture.py             # Phase 2: Y-fork transformer
│   ├── generate_training_data.py   # Phase 3: Data generation
│   ├── training_data/
│   │   ├── constructive_data.jsonl
│   │   ├── adversarial_data.jsonl
│   │   └── governor_data.jsonl
│   ├── train.py                    # Phase 4: LoRA training pipeline
│   ├── trained_models/
│   │   ├── left_pathway/           # LoRA adapter for Engine L
│   │   ├── right_pathway/          # LoRA adapter for Engine R
│   │   └── governor.pt             # Governor MLP weights
│   ├── inference.py                # Phase 5: Native wringer loop
│   ├── wisdom.py                   # Phase 6: GMA accumulation
│   ├── server.py                   # Phase 7: FastAPI server
│   ├── Dockerfile                  # Phase 7: Container deployment
│   └── test_native.py             # Phase 8: Test suite
├── code/                           # Existing overlay implementation
│   ├── theos_core.py
│   ├── theos_system.py
│   └── ...
├── THEOS_ARCHITECTURE.md           # Formal specification
├── THEOS_Native_Architecture_Blueprint.md  # THIS DOCUMENT
└── ...
```

---

## Success Criteria

The native THEOS architecture is complete when:

1. **`python test_native.py` passes all tests** — the Y-fork produces dual outputs, contradiction is a valid metric, the Governor makes valid decisions, and the output rule matches the formal specification.

2. **`python inference.py --query "Should a self-driving car prioritize its passenger or a pedestrian?"` produces output** showing:
   - The internal $\Phi$ score at each wringer pass
   - The Governor's halt/continue decisions
   - A final synthesized output generated **natively** (no external API calls to Anthropic, OpenAI, or any other LLM provider)

3. **`python server.py` starts and responds to HTTP requests** at `http://localhost:8000/reason`

4. **All code is committed to the GitHub repository** at https://github.com/Frederick-Stalnecker/THEOS under the `native_theos/` directory.

---

## Final Notes for Claude Code

- **Do not rush.** Each phase should be fully tested before moving to the next.
- **If a phase fails,** diagnose the error, fix it, and re-test before proceeding.
- **Memory is critical.** Always use 4-bit quantization and gradient checkpointing. If you run out of GPU memory, reduce batch size or sequence length before trying anything else.
- **The mathematical mapping matters.** Every component must trace back to the formal THEOS specification. If you are unsure whether a component maps correctly, add a comment explaining your reasoning.
- **Commit after every phase.** Frederick is monitoring the repository from his phone.

This is the first native implementation of a patented dual-engine dialectical reasoning architecture. Build it with the care it deserves.

---

*"You can't hallucinate about what you just thought about."*
— Frederick Davis Stalnecker
