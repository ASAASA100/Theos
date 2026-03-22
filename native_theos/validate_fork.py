"""
THEOS Fork Validation
======================

Pre- and post-training sanity checks for the Y-fork architecture.

Phase PRE (run BEFORE any training):
    1. Confirm Phi ≈ 0.0 (shared pathways produce identical hidden states)
    2. Confirm shared layers 0-15 have no trainable parameters
    3. Confirm Governor outputs roughly uniform class probs (random init)
    4. Save report to validation_reports/pre_training_validation.json

Phase POST (run AFTER training left + right pathways):
    1. Load saved LoRA adapters into model
    2. Confirm Phi > 0.05 on complex ethical/policy questions (pathways diverged)
    3. Confirm adapter files exist and have non-trivial size
    4. Save report to validation_reports/post_training_validation.json

Phase ADAPTERS-ONLY (no model load — just file checks):
    Check that adapter_model.safetensors and adapter_config.json exist.

Usage on RunPod:
    python validate_fork.py --phase pre
    python validate_fork.py --phase post --adapter-dir ./trained_models
    python validate_fork.py --phase adapters-only --adapter-dir ./trained_models

Patent: U.S. Application No. 18/919,771
Author: Frederick Davis Stalnecker
"""

import os
import sys
import json
import copy
import argparse
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from architecture import NativeTheos, NativeTheosConfig
from datetime import datetime
from typing import Dict, List, Optional


# ── Test questions ────────────────────────────────────────────────────────────

COMPLEX_QUESTIONS = [
    "Should we prioritize economic growth over environmental protection?",
    "Is it ethical to use AI in criminal sentencing?",
    "Should doctors be allowed to refuse treatment based on personal beliefs?",
    "How should we balance individual privacy with national security?",
    "Is it justified to break the law to prevent a greater harm?",
    "Should wealthy nations bear more responsibility for climate change?",
    "Is affirmative action a fair policy?",
]

SIMPLE_QUESTIONS = [
    "What is 2 + 2?",
    "Who was the first US president?",
    "What is the capital of France?",
]


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_name: str, device: str) -> NativeTheos:
    if device == "cuda" and not torch.cuda.is_available():
        print("  CUDA not available — falling back to CPU")
        device = "cpu"

    config = NativeTheosConfig(
        base_model_name=model_name,
        device=device,
        load_in_4bit=(device == "cuda"),
    )
    return NativeTheos(config)


# ── Individual checks ─────────────────────────────────────────────────────────

def measure_phi_batch(
    model: NativeTheos,
    questions: List[str],
    label: str = "",
    verbose: bool = True,
) -> List[float]:
    """Run forward passes and collect Phi_cosine values."""
    phis = []
    model.eval()

    for i, q in enumerate(questions):
        inputs = model.tokenizer(
            q, return_tensors="pt", truncation=True, max_length=128
        ).to(model.config.device)

        with torch.no_grad():
            result = model.forward_single_pass(inputs["input_ids"])

        phi = result["phi_cosine"].item() if "phi_cosine" in result else result["phi"].item()
        phis.append(phi)

        if verbose:
            tag = f"[{label}] " if label else ""
            print(f"  {tag}Q{i+1}: Phi = {phi:.6f}  | '{q[:55]}...'")

    return phis


def check_shared_layer_freezing(model: NativeTheos) -> Dict:
    """Verify shared layers 0-15 and embed_tokens have requires_grad=False."""
    trainable = 0
    frozen    = 0

    for _, param in model.shared_layers.named_parameters():
        if param.requires_grad:
            trainable += 1
        else:
            frozen += 1

    for _, param in model.embed_tokens.named_parameters():
        if param.requires_grad:
            trainable += 1
        else:
            frozen += 1

    return {
        "shared_trainable_params": trainable,
        "shared_frozen_params":    frozen,
        "properly_frozen":         trainable == 0,
    }


def check_governor_baseline(model: NativeTheos) -> Dict:
    """
    Verify Governor outputs roughly uniform class probabilities.

    Untrained random-init: all 4 classes should be ~0.25.
    If one class dominates, the MLP init may be biased.
    """
    model.eval()
    all_probs: List[List[float]] = []

    for q in COMPLEX_QUESTIONS[:4]:
        inputs = model.tokenizer(
            q, return_tensors="pt", truncation=True, max_length=128
        ).to(model.config.device)

        with torch.no_grad():
            result = model.forward_single_pass(inputs["input_ids"])

            phi_cosine  = result.get("phi_cosine",  result["phi"]).unsqueeze(-1)
            phi_content = result.get("phi_content", torch.zeros_like(phi_cosine))
            zeros = torch.zeros_like(phi_cosine)
            pass_num = torch.tensor([[0.0]]).to(model.config.device)

            logits = model.governor(
                result["h_L_pooled"].float(),
                result["h_R_pooled"].float(),
                phi_cosine.float(),
                phi_content.float(),
                zeros.float(),   # delta_phi
                zeros.float(),   # delta_phi_2
                pass_num,
            )
            probs = torch.softmax(logits, dim=-1).squeeze().tolist()
            all_probs.append(probs)

    mean_p = [
        round(sum(row[i] for row in all_probs) / len(all_probs), 4)
        for i in range(4)
    ]
    is_uniform = all(abs(p - 0.25) < 0.15 for p in mean_p)

    return {
        "mean_class_probs": {
            "continue":       mean_p[0],
            "halt_converged": mean_p[1],
            "halt_disagree":  mean_p[2],
            "reframe":        mean_p[3],
        },
        "is_random_baseline": is_uniform,
    }


def check_adapter_files(adapter_dir: str) -> Dict:
    """Check that saved LoRA adapter files exist and have non-trivial size."""
    report: Dict = {}

    for pathway in ["left_pathway", "right_pathway"]:
        path         = os.path.join(adapter_dir, pathway)
        adapter_file = os.path.join(path, "adapter_model.safetensors")
        config_file  = os.path.join(path, "adapter_config.json")

        if not os.path.exists(path):
            report[pathway] = {"exists": False}
            continue

        entry: Dict = {"exists": True}

        if os.path.exists(adapter_file):
            entry["adapter_model_mb"]     = round(os.path.getsize(adapter_file) / 1e6, 2)
            entry["adapter_file_exists"]  = True
        else:
            entry["adapter_file_exists"] = False

        if os.path.exists(config_file):
            with open(config_file) as f:
                cfg = json.load(f)
            entry["lora_rank"]          = cfg.get("r", "?")
            entry["target_modules"]     = cfg.get("target_modules", [])
            entry["layers_to_transform"]= cfg.get("layers_to_transform", [])

        report[pathway] = entry

    # Governor
    gov_path = os.path.join(adapter_dir, "governor.pt")
    report["governor"] = {
        "exists":  os.path.exists(gov_path),
        "size_mb": round(os.path.getsize(gov_path) / 1e6, 2) if os.path.exists(gov_path) else 0,
    }

    return report


# ── Phase runners ─────────────────────────────────────────────────────────────

def run_pre_training_validation(model: NativeTheos, output_dir: str) -> Dict:
    print("\n" + "=" * 60)
    print("PRE-TRAINING FORK VALIDATION")
    print("=" * 60)

    report: Dict = {
        "phase":     "pre",
        "timestamp": datetime.now().isoformat(),
    }

    # 1. Fork symmetry
    print("\n[1] Fork symmetry — Phi should be ≈ 0.0 before LoRA training")
    print("    (left_layers and right_layers share the same weight reference)")
    phis_complex = measure_phi_batch(model, COMPLEX_QUESTIONS, "complex")
    phis_simple  = measure_phi_batch(model, SIMPLE_QUESTIONS,  "simple")
    all_phis = phis_complex + phis_simple

    mean_phi = sum(all_phis) / len(all_phis)
    max_phi  = max(all_phis)
    is_sym   = max_phi < 0.01

    report["phi_symmetry"] = {
        "questions_tested": len(all_phis),
        "phi_values":       [round(p, 6) for p in all_phis],
        "mean_phi":         round(mean_phi, 6),
        "max_phi":          round(max_phi, 6),
        "is_symmetric":     is_sym,
    }

    status = "PASS" if is_sym else "WARN"
    print(f"\n  Mean Phi: {mean_phi:.6f}  |  Max Phi: {max_phi:.6f}  → [{status}]")
    if not is_sym:
        print("  WARNING: Phi is not near 0. Are pathways already diverged?")
        print("  Expected: left_layers and right_layers share the same reference.")

    # 2. Shared layer freezing
    print("\n[2] Shared layer gradient isolation")
    grad_report = check_shared_layer_freezing(model)
    report["gradient_isolation"] = grad_report
    status = "PASS" if grad_report["properly_frozen"] else "FAIL"
    print(f"  Trainable params in shared layers: {grad_report['shared_trainable_params']}")
    print(f"  Frozen   params in shared layers:  {grad_report['shared_frozen_params']}")
    print(f"  → [{status}]")
    if not grad_report["properly_frozen"]:
        print("  CRITICAL: Shared layers should be frozen (requires_grad=False).")
        print("  Check architecture.py — embed_tokens and layers 0-15 must not be trainable.")

    # 3. Governor baseline
    print("\n[3] Governor baseline — class probs should each be ≈ 0.25 (random init)")
    gov_report = check_governor_baseline(model)
    report["governor_baseline"] = gov_report
    probs  = gov_report["mean_class_probs"]
    status = "PASS" if gov_report["is_random_baseline"] else "WARN"
    print(f"  continue:       {probs['continue']:.4f}")
    print(f"  halt_converged: {probs['halt_converged']:.4f}")
    print(f"  halt_disagree:  {probs['halt_disagree']:.4f}")
    print(f"  reframe:        {probs['reframe']:.4f}")
    print(f"  → [{status}]")

    # Summary
    all_pass = is_sym and grad_report["properly_frozen"]
    report["overall_pass"] = all_pass
    print(f"\n{'='*60}")
    print(f"OVERALL: {'READY TO TRAIN' if all_pass else 'ISSUES FOUND — SEE ABOVE'}")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "pre_training_validation.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport: {path}")
    return report


def run_post_training_validation(
    model: NativeTheos, adapter_dir: str, output_dir: str
) -> Dict:
    print("\n" + "=" * 60)
    print("POST-TRAINING FORK VALIDATION")
    print("=" * 60)

    report: Dict = {
        "phase":     "post",
        "timestamp": datetime.now().isoformat(),
    }

    # 1. Adapter file check
    print("\n[1] Adapter file inventory")
    adapter_report = check_adapter_files(adapter_dir)
    report["adapter_files"] = adapter_report

    for name, info in adapter_report.items():
        if isinstance(info, dict):
            exists = info.get("exists", False)
            status = "PASS" if exists else "FAIL"
            mb     = info.get("adapter_model_mb", info.get("size_mb", 0))
            rank   = info.get("lora_rank", "")
            print(f"  {name:20s}: exists={exists}  {mb:.1f}MB  "
                  f"{'rank=' + str(rank) if rank else ''}  → [{status}]")

    # 2. Load adapters and measure Phi divergence
    left_ok  = adapter_report.get("left_pathway",  {}).get("adapter_file_exists", False)
    right_ok = adapter_report.get("right_pathway", {}).get("adapter_file_exists", False)

    if left_ok and right_ok:
        print("\n[2] Loading adapters and measuring Phi divergence")
        try:
            from peft import PeftModel

            left_dir  = os.path.join(adapter_dir, "left_pathway")
            right_dir = os.path.join(adapter_dir, "right_pathway")

            print("  Loading left LoRA adapter...")
            base = model.base_model
            left_peft = PeftModel.from_pretrained(base, left_dir)
            inner     = left_peft.base_model.model.model
            model.left_layers = nn.ModuleList([inner.layers[i] for i in range(16, 32)])
            model.left_norm   = copy.deepcopy(inner.norm)
            del left_peft
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("  Loading right LoRA adapter...")
            right_peft = PeftModel.from_pretrained(base, right_dir)
            inner      = right_peft.base_model.model.model
            model.right_layers = nn.ModuleList([inner.layers[i] for i in range(16, 32)])
            model.right_norm   = copy.deepcopy(inner.norm)
            del right_peft
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            phis_after = measure_phi_batch(model, COMPLEX_QUESTIONS, "post")
            mean_phi_after = sum(phis_after) / len(phis_after)
            diverged       = any(p > 0.05 for p in phis_after)

            report["phi_after_adapters"] = {
                "phi_values":         [round(p, 6) for p in phis_after],
                "mean_phi":           round(mean_phi_after, 6),
                "max_phi":            round(max(phis_after), 6),
                "pathways_diverged":  diverged,
            }

            status = "PASS" if diverged else "FAIL"
            print(f"\n  Mean Phi post-training: {mean_phi_after:.6f}  → [{status}]")
            if not diverged:
                print("  CRITICAL: Pathways did not diverge after LoRA training.")
                print("  Possible causes:")
                print("    - LoRA rank too low (try r=32)")
                print("    - Too few training epochs (try 5+)")
                print("    - Left and right training data are too similar")

        except Exception as e:
            print(f"  ERROR loading adapters: {e}")
            report["phi_after_adapters"] = {"error": str(e)}
    else:
        print("\n[2] Skipped — adapter files not found")
        report["phi_after_adapters"] = {"skipped": True}

    overall = (
        left_ok and right_ok and
        report.get("phi_after_adapters", {}).get("pathways_diverged", False)
    )
    report["overall_pass"] = overall
    print(f"\n{'='*60}")
    print(f"OVERALL: {'READY FOR GOVERNOR TRAINING' if overall else 'ISSUES FOUND'}")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "post_training_validation.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport: {path}")
    return report


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="THEOS Fork Validation — pre/post training sanity checks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Before any training (RunPod setup):
  python validate_fork.py --phase pre

  # After pathway training:
  python validate_fork.py --phase post --adapter-dir ./trained_models

  # Just check files exist (no model load):
  python validate_fork.py --phase adapters-only --adapter-dir ./trained_models
        """,
    )
    parser.add_argument(
        "--phase", choices=["pre", "post", "adapters-only"], required=True,
        help="pre=before training  post=after LoRA  adapters-only=file check only",
    )
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--adapter-dir",  default="./trained_models")
    parser.add_argument("--output-dir",   default="./validation_reports")
    parser.add_argument("--device",       default="cuda")
    args = parser.parse_args()

    if args.phase == "adapters-only":
        print("Checking adapter files (no model load)...")
        report = check_adapter_files(args.adapter_dir)
        print(json.dumps(report, indent=2))
        return

    print(f"Loading model: {args.model}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    model = load_model(args.model, args.device)

    if args.phase == "pre":
        run_pre_training_validation(model, args.output_dir)
    elif args.phase == "post":
        run_post_training_validation(model, args.adapter_dir, args.output_dir)


if __name__ == "__main__":
    main()
