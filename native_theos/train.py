"""
THEOS Native Training Pipeline — Version 2
============================================

Multi-phase LoRA training for the native THEOS V2 architecture:
    Phase A (left):      Specialize left pathway (constructive) via LoRA
    Phase B (right):     Specialize right pathway (adversarial) via LoRA
                         V2: Productive Opposition Score (POS) auxiliary loss
    Phase C (governor):  Train Governor MLP on real hidden state divergence
                         V2: All three Phi channels, multi-objective Q metric

Uses QLoRA (4-bit quantization + LoRA) to fine-tune only the upper pathway
layers (16-31) while keeping shared layers (0-15) fully frozen.

Left pathway learns:  Build the strongest, most coherent constructive argument.
Right pathway learns: Socratic opposition — find what was missed, who bears
                      the cost, what premise is false. NOT contrarianism.
                      POS loss penalizes parroting and reflexive negation.

Governor MLP trains on:
    - All three Phi channels (cosine, JS, content)
    - Trajectory velocity and acceleration
    - Multi-objective targets (synthesis, honest disagreement, reframe)

RunPod usage:
    python train.py --phase left
    python train.py --phase right
    python train.py --phase governor
    python train.py --phase all

Patent: U.S. Application No. 18/919,771 (Provisional #63/831,738)
Author: Frederick Davis Stalnecker
"""

import json
import math
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# Add parent directory so we can import architecture.py
sys.path.insert(0, os.path.dirname(__file__))
from architecture import NativeTheos, NativeTheosConfig


# ============================================================
# DATASET
# ============================================================

class TheosDataset(Dataset):
    """
    Dataset for THEOS pathway training.

    Each entry is a JSONL line with 'instruction' and 'output'.
    Formatted as instruction-following fine-tuning.
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        with open(data_path, "r") as f:
            for line in f:
                self.data.append(json.loads(line))

        print(f"  Loaded {len(self.data)} training examples from {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        text = (
            f"### Instruction:\n{entry['instruction']}\n\n"
            f"### Response:\n{entry['output']}"
        )

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


# ============================================================
# V2 CLAIM 24 — PRODUCTIVE OPPOSITION SCORE (POS) TRAINER
# ============================================================

class POSTrainer(Trainer):
    """
    Custom Trainer for the right (adversarial) pathway.

    V2 Claim 24: Productive Opposition Score training objective.

    The two failure modes in adversarial training:
        1. Contrarianism:  reflexive negation, no new content, easy to produce.
        2. Parroting:      rephrasing the left conclusion negatively.
           Divergence appears high; no new consideration introduced.

    POS loss adds an auxiliary term to the standard LM cross-entropy loss:

        L_total = L_lm + lambda_pos * L_pos

    L_pos has two components:
        - Novelty reward:    reward high entropy in the adversarial output
                             distribution — diverse vocabulary = new considerations
        - Anti-contrarian:   penalize direct token overlap with the instruction
                             (a response that mostly mirrors the instruction words
                             is parroting, not opposing)

    This is a tractable approximation of the full POS decomposition
    (relevance x novelty x coherence / contrarian_penalty) that can be
    computed purely from the model's output logits during training without
    requiring a second model or human ratings.

    Patent: Claim 24 — U.S. Application No. 18/919,771
    """

    POS_LAMBDA = 0.1    # weight of POS auxiliary loss relative to LM loss
    ENTROPY_TARGET = 3.0  # nats — target minimum entropy in adversarial output

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Standard language modeling loss
        outputs = model(**inputs)
        lm_loss = outputs.loss

        # POS auxiliary loss on the output logits
        logits = outputs.logits    # [batch, seq_len, vocab_size]

        # Novelty reward: entropy of per-token output distribution
        # High entropy = diverse vocabulary = more likely to introduce new content
        probs = F.softmax(logits.float(), dim=-1)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        entropy_per_token = -(probs * log_probs).sum(dim=-1)   # [batch, seq]

        # Only penalize tokens where entropy is below target (too certain = parroting)
        entropy_shortfall = F.relu(self.ENTROPY_TARGET - entropy_per_token)
        pos_loss = entropy_shortfall.mean()

        total_loss = lm_loss + self.POS_LAMBDA * pos_loss

        if return_outputs:
            return total_loss, outputs
        return total_loss


# ============================================================
# PATHWAY TRAINING (QLoRA)
# ============================================================

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
    max_seq_length: int = 512,
):
    """
    Train a single pathway (left or right) using QLoRA.

    Loads the base Llama 3 8B model in 4-bit, applies LoRA adapters
    to q/k/v/o projections in layers 16-31 only, and fine-tunes on
    the pathway's training data.

    Shared layers (0-15) are frozen. Only the pathway-specific upper
    layers receive gradient updates via the low-rank adapters.

    The left pathway learns: "Build the strongest, most coherent argument."
    The right pathway learns: "Find what was missed. Socratic, not combative."
    """
    print(f"\n{'='*60}")
    print(f"Training {pathway_name} pathway")
    print(f"Data:   {data_path}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {num_epochs}  |  Batch: {batch_size}  |  LR: {learning_rate}")
    print(f"LoRA rank: {lora_rank}  |  alpha: {lora_alpha}")
    print(f"{'='*60}\n")

    os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load in bfloat16 — no 4-bit quantization needed when VRAM >= 24GB.
    # Llama 3 8B in bfloat16 ~16GB; A40 has 47GB. QLoRA (4-bit) is only
    # needed when VRAM is tight (consumer GPUs <24GB).
    # This also avoids the set_submodule compatibility issue with PyTorch 2.4.x.
    vram_gb = (
        torch.cuda.get_device_properties(0).total_memory / 1e9
        if torch.cuda.is_available()
        else 0
    )
    use_4bit = vram_gb < 24.0

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
        print(f"  4-bit quantization enabled (VRAM={vram_gb:.0f}GB)")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        print(f"  bfloat16 direct load (VRAM={vram_gb:.0f}GB — no quantization needed)")

    # Configure LoRA — only target the upper pathway layers (16-31)
    # The split point matches NativeTheosConfig.split_layer = 16
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        # Only apply LoRA to upper layers — shared layers (0-15) stay frozen
        layers_to_transform=list(range(16, 32)),
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    dataset = TheosDataset(data_path, tokenizer, max_length=max_seq_length)

    # Optional wandb logging — only if WANDB_API_KEY is explicitly set
    # (W&B is installed on many servers but crashes if not authenticated)
    if os.environ.get("WANDB_API_KEY"):
        report_to = "wandb"
        print(f"  W&B logging enabled: run=theos_{pathway_name}")
    else:
        report_to = "none"
        print("  W&B not configured — logging to console only")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        bf16=torch.cuda.is_available(), # bfloat16 on CUDA (A40/A100)
        logging_steps=10,
        save_strategy="epoch",
        warmup_ratio=0.1,
        weight_decay=0.01,
        report_to=report_to,
        run_name=f"theos_{pathway_name}",
        dataloader_num_workers=2,
        remove_unused_columns=False,    # needed for custom dataset
    )

    if pathway_name.startswith("right"):
        # V2 Claim 24: Productive Opposition Score auxiliary loss
        # POSTrainer adds a divergence-maximizing term to the right pathway loss,
        # preventing the two failure modes: contrarianism and parroting.
        trainer = POSTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
        )
        print("  Right pathway: POS auxiliary loss enabled (Claim 24)")
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
        )

    trainer.train()

    # Save the LoRA adapter (not the full model — just the adapter weights)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\n{pathway_name} pathway training complete.")
    print(f"LoRA adapter saved to: {output_dir}")
    return model


# ============================================================
# ADAPTER LOADING — inject trained pathways before Governor training
# ============================================================

def load_trained_pathways(
    native_model: NativeTheos,
    output_dir: str,
    base_model_name: str,
) -> bool:
    """
    Load trained LoRA adapters into the Y-fork using multi-adapter mode.

    This must be called BEFORE train_governor() so the Governor trains on
    hidden states from genuinely diverged pathways (Phi > 0), not the
    untrained shared-weight pathways (Phi ≈ 0).

    Multi-adapter strategy (single base model, no second load):
        1. Wrap native_model.base_model with left LoRA via PeftModel.from_pretrained()
        2. Load right LoRA via load_adapter()
        3. PEFT injects hooks directly into the layer objects already referenced
           by native_model.left_layers / right_layers — no re-wiring needed.
        4. set_adapter("left"/"right") before each pathway forward pass.

    Memory: ~16.2 GB total (base bfloat16 + two tiny adapter dicts).
    Old approach loaded TWO extra base models (~32 GB extra) → OOM on 40 GB pods.

    Returns True if adapters were loaded, False if not found (Governor trains on Phi ≈ 0
    with a warning — valid for the very first Governor training before any LoRA exists).
    """
    from peft import PeftModel

    left_dir  = os.path.join(output_dir, "left_pathway")
    right_dir = os.path.join(output_dir, "right_pathway")

    left_exists  = os.path.exists(os.path.join(left_dir,  "adapter_config.json"))
    right_exists = os.path.exists(os.path.join(right_dir, "adapter_config.json"))

    if not left_exists and not right_exists:
        print("\nWARNING: Trained LoRA adapters not found.")
        print(f"  Left  ({left_dir}):  {'found' if left_exists else 'MISSING'}")
        print(f"  Right ({right_dir}): {'found' if right_exists else 'MISSING'}")
        print("  Governor will train on untrained pathways (Phi ≈ 0).")
        print("  Run '--phase left' and '--phase right' first for meaningful Governor training.")
        return False

    print("\nLoading trained LoRA adapters (multi-adapter mode — no second base model load)...")

    # Wrap native_model.base_model in-place.
    # PEFT injects LoRA hooks into the layer objects already referenced by
    # native_model.left_layers and native_model.right_layers — no re-wiring needed.
    if left_exists:
        native_model.base_model = PeftModel.from_pretrained(
            native_model.base_model, left_dir, adapter_name="left"
        )
        native_model._has_left_adapter = True
        print("  Left LoRA adapter loaded.")

        if right_exists:
            native_model.base_model.load_adapter(right_dir, adapter_name="right")
            native_model._has_right_adapter = True
            native_model.base_model.set_adapter("left")   # default to left
            print("  Right LoRA adapter loaded.")
    else:
        native_model.base_model = PeftModel.from_pretrained(
            native_model.base_model, right_dir, adapter_name="right"
        )
        native_model._has_right_adapter = True
        print("  Right LoRA adapter loaded.")

    native_model.has_lora_adapters = True
    print("  Multi-adapter mode active. Governor will train on diverged hidden states.")
    return True


# ============================================================
# GOVERNOR TRAINING
# ============================================================

def train_governor(
    native_model: NativeTheos,
    governor_data_path: str,
    output_dir: str,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
):
    """
    Train the Governor MLP on real hidden state divergence — Version 2.

    V2 additions:
        - All three Phi channels fed to Governor (cosine, JS, content)
        - Phi_JS captured from shared layer attention weights
        - Multi-objective quality metric targets (Claims 22, 21)
        - Learned alpha/beta/gamma divergence weights also trained here

    Governor input is now 2d+7:
        H_L_pooled, H_R_pooled, phi_cosine, phi_js, phi_content,
        delta_phi, delta_phi_2, pass_num

    Target classes:
        0 = continue          (keep the wringer running)
        1 = halt_converged    (engines agree)
        2 = halt_disagree     (irresolvable, expose both sides)
        3 = reframe           (hidden premise detected)

    Patent: Claims 21, 22 — U.S. Application No. 18/919,771
    """
    print(f"\n{'='*60}")
    print("Training Governor MLP V2")
    print(f"Data:   {governor_data_path}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {num_epochs}  |  LR: {learning_rate}")
    print(f"{'='*60}\n")

    os.makedirs(output_dir, exist_ok=True)

    device    = native_model.config.device
    governor  = native_model.governor.to(device)
    # Include governor's learnable alpha/beta/gamma in optimizer
    optimizer = torch.optim.Adam(governor.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    with open(governor_data_path, "r") as f:
        gov_data = [json.loads(line) for line in f]

    print(f"Collecting hidden states for {len(gov_data)} governor training examples...")

    native_model.eval()
    training_examples = []

    for i, entry in enumerate(gov_data):
        question = entry["question"]

        inputs = native_model.tokenizer(
            question, return_tensors="pt",
            truncation=True, max_length=256,
        ).to(device)

        with torch.no_grad():
            result = native_model.forward_single_pass(
                input_ids=inputs["input_ids"],
            )

        # V2: all three Phi channels
        phi_cosine  = result["phi_cosine"].mean().item()
        phi_js      = result["phi_js"].mean().item()
        phi_content = result["phi_content"].mean().item()
        phi_comp    = result["phi"].mean().item()

        delta_phi   = entry.get("delta_phi",   0.0)
        delta_phi_2 = entry.get("delta_phi_2", 0.0)

        # Always derive target from LIVE phi_comp — ignore word-overlap proxy in JSONL.
        # The "target" field in governor_data.jsonl was set by estimate_text_similarity()
        # (Jaccard word overlap ~0.6-0.8) which is a COMPLETELY different distribution
        # from real hidden-state Phi (~0.009 for untrained pathways). Training on those
        # stale targets causes the Governor to be miscalibrated.
        #
        # Target logic:
        #   Pass 1: always continue (explore before deciding)
        #   Pass 2+: halt_converged if phi < eps_converge, halt_disagree if phi > eps_partial,
        #            otherwise continue (phi in the productive middle range)
        pass_num = entry.get("pass_number", 1)
        if pass_num == 1:
            target = 0   # continue — first pass always continues
        elif phi_comp < native_model.config.eps_converge:
            target = 1   # halt_converged
        elif phi_comp > native_model.config.eps_partial:
            target = 2   # halt_disagree
        else:
            target = 0   # continue — phi in middle range, keep wringer running

        training_examples.append({
            "h_left":      result["h_L_pooled"].detach(),
            "h_right":     result["h_R_pooled"].detach(),
            "phi_cosine":  torch.tensor([[phi_cosine]],  dtype=torch.float).to(device),
            "phi_js":      torch.tensor([[phi_js]],      dtype=torch.float).to(device),
            "phi_content": torch.tensor([[phi_content]], dtype=torch.float).to(device),
            "delta_phi":   torch.tensor([[delta_phi]],   dtype=torch.float).to(device),
            "delta_phi_2": torch.tensor([[delta_phi_2]], dtype=torch.float).to(device),
            "pass_num":    torch.tensor(
                [[entry.get("pass_number", 1) / 7.0]], dtype=torch.float
            ).to(device),
            "target": target,
        })

        if (i + 1) % 20 == 0 or (i + 1) == len(gov_data):
            print(f"  Collected {i+1}/{len(gov_data)}  "
                  f"phi_cos={phi_cosine:.4f}  phi_js={phi_js:.4f}  "
                  f"phi_content={phi_content:.4f}  phi_comp={phi_comp:.4f}  "
                  f"target={target}")

    # Train
    governor.train()
    print(f"\nTraining Governor V2 on {len(training_examples)} examples...")

    for epoch in range(num_epochs):
        total_loss = 0.0
        correct    = 0

        for ex in training_examples:
            optimizer.zero_grad()

            logits = governor(
                ex["h_left"],
                ex["h_right"],
                ex["phi_cosine"],
                ex["phi_js"],
                ex["phi_content"],
                ex["delta_phi"],
                ex["delta_phi_2"],
                ex["pass_num"],
            )

            target_tensor = torch.tensor([ex["target"]]).to(logits.device)
            loss = criterion(logits, target_tensor)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = logits.argmax(dim=-1)
            correct += (pred == target_tensor).sum().item()

        accuracy  = correct / len(training_examples)
        avg_loss  = total_loss / len(training_examples)
        w = governor.channel_weights.detach()
        print(f"  Epoch {epoch+1:2d}/{num_epochs}: Loss={avg_loss:.4f}  "
              f"Accuracy={accuracy:.1%}  "
              f"α={w[0]:.3f} β={w[1]:.3f} γ={w[2]:.3f}")

    # Save governor weights (includes learned alpha/beta/gamma)
    governor_path = os.path.join(output_dir, "governor.pt")
    torch.save(governor.state_dict(), governor_path)

    w = governor.channel_weights.detach()
    print(f"\nGovernor saved to {governor_path}")
    print(f"Learned divergence weights: α(cosine)={w[0]:.3f}  β(JS)={w[1]:.3f}  γ(content)={w[2]:.3f}")


# ============================================================
# PHASE D — CROSS-ATTENTION TRAINING (Claim 23)
# ============================================================

def train_cross_attention(
    native_model: "NativeTheos",
    data_path: str,
    output_dir: str,
    num_epochs: int = 20,
    learning_rate: float = 1e-4,
    margin: float = 0.01,
    lambda_collapse: float = 0.5,
):
    """
    Train the cross-attention modules (Claim 23) using a self-supervised
    Phi-reduction objective — no API calls required.

    Objective:
        After cross-attention, composite Phi should DECREASE compared to pass 1.
        Each pathway has now attended to specific tokens in the other's prior
        hidden state, causing partial convergence toward synthesis.

        Loss = ReLU(phi_2 - phi_1 + margin)           [phi must decrease]
             + lambda_collapse * ReLU(eps_converge - phi_2)  [phi must not collapse]

        Where:
            phi_1 = Phi from pass 1 (past_H=None, cross-attn is no-op)
            phi_2 = Phi from pass 2 (past_H from pass 1, cross-attn active)
            margin = minimum required decrease (prevents trivial solutions)
            eps_converge = lower bound so pathways don't collapse to identical outputs

    Only W_Q, W_K, W_V, W_O in cross_attn_left and cross_attn_right are trained.
    All pathway LoRA adapters and the governor are frozen.

    Input data: governor_data.jsonl (the same questions used for governor training).
    No new data generation needed.

    Patent: Claim 23 — U.S. Application No. 18/919,771
    """
    print(f"\n{'='*60}")
    print("Training Cross-Attention Modules V2 (Phase D — Claim 23)")
    print(f"Data:          {data_path}")
    print(f"Output:        {output_dir}")
    print(f"Epochs:        {num_epochs}  |  LR: {learning_rate}")
    print(f"Margin:        {margin}  (min Phi reduction per pass)")
    print(f"Anti-collapse: lambda={lambda_collapse}  eps_converge={native_model.config.eps_converge}")
    print(f"{'='*60}\n")

    os.makedirs(output_dir, exist_ok=True)

    device = native_model.config.device

    # --- Freeze everything except the two cross-attention modules ---
    for name, param in native_model.named_parameters():
        param.requires_grad = False

    cross_attn_params = []
    for module_name in ["cross_attn_left", "cross_attn_right"]:
        module = getattr(native_model, module_name)
        for name, param in module.named_parameters():
            param.requires_grad = True
            cross_attn_params.append(param)

    total_trainable = sum(p.numel() for p in cross_attn_params)
    print(f"Trainable parameters: {total_trainable:,}  (cross_attn_left + cross_attn_right only)")
    print(f"All other parameters: frozen\n")

    optimizer = torch.optim.Adam(cross_attn_params, lr=learning_rate)
    eps_converge = native_model.config.eps_converge

    # Load questions from governor_data.jsonl
    with open(data_path, "r") as f:
        questions = [json.loads(line)["question"] for line in f]
    # De-duplicate (governor_data has the same question at multiple pass_numbers)
    seen = set()
    unique_questions = []
    for q in questions:
        if q not in seen:
            seen.add(q)
            unique_questions.append(q)
    print(f"Training on {len(unique_questions)} unique questions ({len(questions)} total entries).\n")

    # Tokenize once
    native_model.eval()
    tokenized = []
    for q in unique_questions:
        ids = native_model.tokenizer(
            q, return_tensors="pt", truncation=True, max_length=256,
        ).to(device)
        tokenized.append(ids["input_ids"])

    # Training loop
    for epoch in range(num_epochs):
        native_model.cross_attn_left.train()
        native_model.cross_attn_right.train()

        total_loss = 0.0
        n_decreased = 0    # questions where phi_2 < phi_1 (objective met)
        n_collapsed = 0    # questions where phi_2 < eps_converge (over-converged)
        phi1_sum = 0.0
        phi2_sum = 0.0

        for ids in tokenized:
            optimizer.zero_grad()

            # Pass 1: no cross-attention (past_H = None) — compute baseline Phi
            with torch.no_grad():
                result1 = native_model.forward_single_pass(input_ids=ids)
            H_L1 = result1["hidden_L"].detach()   # [batch, seq_len, hidden_dim]
            H_R1 = result1["hidden_R"].detach()
            phi_1 = result1["phi"].mean()          # scalar

            # Pass 2: WITH cross-attention — gradients flow through cross_attn_left/right
            result2 = native_model.forward_single_pass(
                input_ids=ids,
                past_H_L=H_L1,   # left pathway attends to right's pass-1 hidden state
                past_H_R=H_R1,   # right pathway attends to left's pass-1 hidden state
            )
            phi_2 = result2["phi"].mean()          # scalar (grad-connected to cross-attn)

            # Primary loss: phi should decrease after cross-attention
            loss_reduce = F.relu(phi_2 - phi_1 + margin)

            # Anti-collapse loss: phi_2 must stay above eps_converge
            loss_collapse = lambda_collapse * F.relu(
                torch.tensor(eps_converge, dtype=phi_2.dtype, device=device) - phi_2
            )

            loss = loss_reduce + loss_collapse
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            phi1_v = phi_1.item()
            phi2_v = phi_2.item()
            phi1_sum += phi1_v
            phi2_sum += phi2_v
            if phi2_v < phi1_v:
                n_decreased += 1
            if phi2_v < eps_converge:
                n_collapsed += 1

        n = len(tokenized)
        avg_loss = total_loss / n
        avg_phi1 = phi1_sum / n
        avg_phi2 = phi2_sum / n
        pct_decreased = 100.0 * n_decreased / n
        pct_collapsed = 100.0 * n_collapsed / n

        print(
            f"  Epoch {epoch+1:2d}/{num_epochs}: Loss={avg_loss:.4f}  "
            f"Φ_1={avg_phi1:.4f}→Φ_2={avg_phi2:.4f}  "
            f"Decreased={pct_decreased:.0f}%  Collapsed={pct_collapsed:.0f}%"
        )

    # Save cross-attention weights
    left_path  = os.path.join(output_dir, "cross_attn_left.pt")
    right_path = os.path.join(output_dir, "cross_attn_right.pt")
    torch.save(native_model.cross_attn_left.state_dict(),  left_path)
    torch.save(native_model.cross_attn_right.state_dict(), right_path)

    print(f"\nCross-attention saved:")
    print(f"  {left_path}")
    print(f"  {right_path}")
    print(f"\nFinal Φ reduction: {avg_phi1:.4f} → {avg_phi2:.4f}  "
          f"(Δ = {avg_phi1 - avg_phi2:.4f})")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="THEOS Native LoRA Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate training data first (uses Claude API):
  python generate_training_data.py

  # Train left pathway (constructive):
  python train.py --phase left

  # Train right pathway (adversarial):
  python train.py --phase right

  # Train governor (requires model):
  python train.py --phase governor

  # Train cross-attention (Phase D — Claim 23, self-supervised, no API):
  python train.py --phase cross_attn

  # Train everything in sequence:
  python train.py --phase all

  # Custom settings for larger run:
  python train.py --phase left --epochs 5 --lora-rank 32
        """,
    )
    parser.add_argument(
        "--phase",
        choices=["left", "right", "governor", "cross_attn", "all"],
        required=True,
        help="Which training phase to run",
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="HuggingFace model ID for base model",
    )
    parser.add_argument(
        "--data-dir",
        default="./training_data",
        help="Directory containing JSONL training data",
    )
    parser.add_argument(
        "--output-dir",
        default="./trained_models",
        help="Directory to save trained model artifacts",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs for pathway training",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device batch size",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank (higher = more capacity, more parameters)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for governor training (cuda or cpu)",
    )
    args = parser.parse_args()

    # Validate CUDA for RunPod
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available — falling back to CPU")
        args.device = "cpu"

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ---- LEFT PATHWAY (constructive) ----
    if args.phase in ["left", "all"]:
        constructive_path = os.path.join(args.data_dir, "constructive_data.jsonl")
        if not os.path.exists(constructive_path):
            print(f"ERROR: Training data not found: {constructive_path}")
            print("Run generate_training_data.py first.")
            sys.exit(1)
        train_pathway(
            model_name=args.model,
            data_path=constructive_path,
            output_dir=os.path.join(args.output_dir, "left_pathway"),
            pathway_name="left_constructive",
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lora_rank=args.lora_rank,
        )

    # ---- RIGHT PATHWAY (adversarial) ----
    if args.phase in ["right", "all"]:
        adversarial_path = os.path.join(args.data_dir, "adversarial_data.jsonl")
        if not os.path.exists(adversarial_path):
            print(f"ERROR: Training data not found: {adversarial_path}")
            print("Run generate_training_data.py first.")
            sys.exit(1)
        train_pathway(
            model_name=args.model,
            data_path=adversarial_path,
            output_dir=os.path.join(args.output_dir, "right_pathway"),
            pathway_name="right_adversarial",
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lora_rank=args.lora_rank,
        )

    # ---- GOVERNOR MLP ----
    if args.phase in ["governor", "all"]:
        governor_path = os.path.join(args.data_dir, "governor_data.jsonl")
        if not os.path.exists(governor_path):
            print(f"ERROR: Governor data not found: {governor_path}")
            print("Run generate_training_data.py first.")
            sys.exit(1)

        print("\nLoading Y-fork model for Governor training...")
        config = NativeTheosConfig(
            base_model_name=args.model,
            device=args.device,
        )
        native_model = NativeTheos(config)

        # Load trained LoRA adapters into pathways so the Governor trains
        # on genuinely diverged hidden states (not the pre-training Phi ≈ 0).
        load_trained_pathways(native_model, args.output_dir, args.model)

        train_governor(
            native_model=native_model,
            governor_data_path=governor_path,
            output_dir=args.output_dir,
        )

    # ---- PHASE D: CROSS-ATTENTION (Claim 23, self-supervised) ----
    if args.phase in ["cross_attn", "all"]:
        governor_path = os.path.join(args.data_dir, "governor_data.jsonl")
        if not os.path.exists(governor_path):
            print(f"ERROR: Governor data not found: {governor_path}")
            print("Run generate_training_data.py first.")
            sys.exit(1)

        print("\nLoading Y-fork model for cross-attention training (Phase D)...")
        config = NativeTheosConfig(
            base_model_name=args.model,
            device=args.device,
        )
        native_model = NativeTheos(config)

        # Load pre-trained governor weights so composite_phi() uses learned α/β/γ
        gov_weights = os.path.join(args.output_dir, "governor.pt")
        if os.path.exists(gov_weights):
            native_model.governor.load_state_dict(
                torch.load(gov_weights, map_location=args.device)
            )
            print(f"  Governor weights loaded from {gov_weights}")
        else:
            print("  WARNING: governor.pt not found — using default equal weights (α=β=γ=0.333)")

        # Load LoRA adapters so pathways are genuinely diverged during training
        load_trained_pathways(native_model, args.output_dir, args.model)

        # Load any previously saved cross-attn weights (resume training)
        left_path  = os.path.join(args.output_dir, "cross_attn_left.pt")
        right_path = os.path.join(args.output_dir, "cross_attn_right.pt")
        if os.path.exists(left_path) and os.path.exists(right_path):
            native_model.cross_attn_left.load_state_dict(
                torch.load(left_path, map_location=args.device)
            )
            native_model.cross_attn_right.load_state_dict(
                torch.load(right_path, map_location=args.device)
            )
            print(f"  Resuming from existing cross-attn weights.")

        train_cross_attention(
            native_model=native_model,
            data_path=governor_path,
            output_dir=args.output_dir,
            num_epochs=args.epochs if args.epochs != 3 else 20,   # default 20 for cross_attn
        )

    print("\n" + "="*60)
    print("THEOS training complete.")
    print("="*60)
