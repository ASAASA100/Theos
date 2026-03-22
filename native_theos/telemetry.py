"""
THEOS Training Telemetry
=========================

Logs all training metrics to CSV files and console dashboard.

Three loggers:
    PathwayTrainingCallback — HuggingFace Trainer callback for left/right pathway training.
        Logs per-step: step, loss, learning_rate, grad_norm
        Logs per-epoch: epoch, train_loss, epoch_time

    GovernorTrainingLogger — Used inside train_governor() loop.
        Logs per-epoch: epoch, loss, accuracy,
                        mean_phi_cosine, mean_phi_content,
                        class_distribution (continue/converged/disagree/reframe)

    WringerTelemetry — Used inside run_native_wringer() for per-query analysis.
        Logs per-pass: query_id, pass_num, phi_cosine, phi_content,
                       delta_phi, delta_phi_2, governor_decision

    CheckpointLogger — Called after each checkpoint to log LoRA weight norms.
        Logs per-checkpoint: step, layer, lora_A_norm, lora_B_norm, delta_norm

Usage in train.py:
    from telemetry import PathwayTrainingCallback, GovernorTrainingLogger

    telemetry = PathwayTrainingCallback(pathway_name="left", output_dir="./logs")
    trainer = Trainer(..., callbacks=[telemetry])

Usage in inference.py:
    from telemetry import WringerTelemetry
    wringer_log = WringerTelemetry(output_dir="./logs")
    # call wringer_log.log_pass(...) inside the wringer loop

Patent: U.S. Application No. 18/919,771
Author: Frederick Davis Stalnecker
"""

import os
import csv
import time
import json
import torch
import torch.nn as nn
from datetime import datetime
from typing import Dict, List, Optional, Any


# ── Utilities ─────────────────────────────────────────────────────────────────

def _ensure_csv(path: str, fieldnames: List[str]) -> None:
    """Create CSV with header if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()


def _append_csv(path: str, fieldnames: List[str], row: Dict) -> None:
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writerow(row)


# ── Pathway Training Callback ─────────────────────────────────────────────────

try:
    from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

    class PathwayTrainingCallback(TrainerCallback):
        """
        HuggingFace Trainer callback for left/right pathway LoRA training.

        Attach to Trainer(..., callbacks=[PathwayTrainingCallback(...)]).

        CSV outputs:
            {output_dir}/{pathway_name}_steps.csv   — per-step metrics
            {output_dir}/{pathway_name}_epochs.csv  — per-epoch summary
        """

        STEP_FIELDS  = ["step", "loss", "learning_rate", "grad_norm", "timestamp"]
        EPOCH_FIELDS = ["epoch", "train_loss", "epoch_time_s", "timestamp"]

        def __init__(self, pathway_name: str, output_dir: str = "./logs"):
            self.pathway_name = pathway_name
            self.output_dir   = output_dir

            self.step_csv  = os.path.join(output_dir, f"{pathway_name}_steps.csv")
            self.epoch_csv = os.path.join(output_dir, f"{pathway_name}_epochs.csv")

            _ensure_csv(self.step_csv,  self.STEP_FIELDS)
            _ensure_csv(self.epoch_csv, self.EPOCH_FIELDS)

            self._epoch_start_time: float = time.time()
            self._epoch_losses: List[float] = []

        def on_log(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            logs: Optional[Dict[str, Any]] = None,
            **kwargs,
        ):
            if logs is None:
                return

            loss = logs.get("loss")
            lr   = logs.get("learning_rate")
            grad = logs.get("grad_norm")

            if loss is not None:
                self._epoch_losses.append(float(loss))
                row = {
                    "step":          state.global_step,
                    "loss":          round(float(loss), 6),
                    "learning_rate": round(float(lr), 8) if lr is not None else "",
                    "grad_norm":     round(float(grad), 4) if grad is not None else "",
                    "timestamp":     datetime.now().isoformat(),
                }
                _append_csv(self.step_csv, self.STEP_FIELDS, row)

                # Console dashboard (every 10 steps)
                if state.global_step % 10 == 0:
                    print(
                        f"  [{self.pathway_name}] step={state.global_step:5d}  "
                        f"loss={loss:.4f}  lr={lr:.2e}  "
                        f"grad={grad:.4f}" if grad else f"  [{self.pathway_name}] "
                        f"step={state.global_step:5d}  loss={loss:.4f}"
                    )

        def on_epoch_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ):
            epoch_time = time.time() - self._epoch_start_time
            mean_loss  = (
                sum(self._epoch_losses) / len(self._epoch_losses)
                if self._epoch_losses else 0.0
            )

            row = {
                "epoch":        state.epoch,
                "train_loss":   round(mean_loss, 6),
                "epoch_time_s": round(epoch_time, 1),
                "timestamp":    datetime.now().isoformat(),
            }
            _append_csv(self.epoch_csv, self.EPOCH_FIELDS, row)

            print(
                f"\n  [{self.pathway_name}] EPOCH {state.epoch:.0f} complete  "
                f"mean_loss={mean_loss:.4f}  time={epoch_time:.0f}s\n"
            )

            self._epoch_start_time = time.time()
            self._epoch_losses.clear()

except ImportError:
    # transformers not available (e.g., running tests locally without GPU deps)
    class PathwayTrainingCallback:  # type: ignore
        def __init__(self, *args, **kwargs):
            print("WARNING: transformers not installed — PathwayTrainingCallback disabled")


# ── Governor Training Logger ──────────────────────────────────────────────────

class GovernorTrainingLogger:
    """
    Logger for the Governor MLP training loop.

    Call log_epoch() at the end of each epoch inside train_governor().

    CSV output: {output_dir}/governor_training.csv
    """

    FIELDS = [
        "epoch", "loss", "accuracy",
        "mean_phi_cosine", "mean_phi_content",
        "p_continue", "p_converged", "p_disagree", "p_reframe",
        "timestamp",
    ]

    def __init__(self, output_dir: str = "./logs"):
        self.output_dir = output_dir
        self.csv_path   = os.path.join(output_dir, "governor_training.csv")
        _ensure_csv(self.csv_path, self.FIELDS)

    def log_epoch(
        self,
        epoch: int,
        loss: float,
        accuracy: float,
        training_examples: List[Dict],
        governor: nn.Module,
    ) -> None:
        """
        Log one epoch of Governor training.

        Computes mean Phi values and class prediction distribution
        from the training examples for that epoch.
        """
        governor.eval()

        phi_cosines:  List[float] = []
        phi_contents: List[float] = []
        class_counts  = [0, 0, 0, 0]  # continue, converged, disagree, reframe

        with torch.no_grad():
            for ex in training_examples:
                phi_cosines.append(ex["phi_cosine"].item())
                phi_contents.append(ex["phi_content"].item())

                logits = governor(
                    ex["h_left"],
                    ex["h_right"],
                    ex["phi_cosine"],
                    ex["phi_content"],
                    ex["delta_phi"],
                    ex["delta_phi_2"],
                    ex["pass_num"],
                )
                pred = logits.argmax(dim=-1).item()
                class_counts[pred] += 1

        n = len(training_examples) or 1
        row = {
            "epoch":            epoch,
            "loss":             round(loss, 6),
            "accuracy":         round(accuracy, 4),
            "mean_phi_cosine":  round(sum(phi_cosines)  / n, 6),
            "mean_phi_content": round(sum(phi_contents) / n, 6),
            "p_continue":       round(class_counts[0] / n, 4),
            "p_converged":      round(class_counts[1] / n, 4),
            "p_disagree":       round(class_counts[2] / n, 4),
            "p_reframe":        round(class_counts[3] / n, 4),
            "timestamp":        datetime.now().isoformat(),
        }
        _append_csv(self.csv_path, self.FIELDS, row)

        # Console dashboard
        print(
            f"  [governor] epoch={epoch:3d}  loss={loss:.4f}  acc={accuracy:.1%}  "
            f"Φ_cos={row['mean_phi_cosine']:.4f}  Φ_content={row['mean_phi_content']:.4f}  "
            f"decisions=[{class_counts[0]}C/{class_counts[1]}CV/"
            f"{class_counts[2]}D/{class_counts[3]}R]"
        )

        governor.train()


# ── Wringer Inference Telemetry ────────────────────────────────────────────────

class WringerTelemetry:
    """
    Per-query, per-pass logging for run_native_wringer().

    Attach to inference to track Phi trajectory, Governor decisions,
    and mode routing across queries.

    CSV output: {output_dir}/wringer_passes.csv
    """

    FIELDS = [
        "query_id", "query_preview", "mode",
        "pass_num", "phi_cosine", "phi_content",
        "delta_phi", "delta_phi_2",
        "governor_decision", "halt_reason",
        "passes_used", "timestamp",
    ]

    def __init__(self, output_dir: str = "./logs"):
        self.output_dir = output_dir
        self.csv_path   = os.path.join(output_dir, "wringer_passes.csv")
        _ensure_csv(self.csv_path, self.FIELDS)
        self._query_counter = 0

    def new_query(self, query: str, mode: int) -> int:
        """Register a new query. Returns query_id."""
        self._query_counter += 1
        return self._query_counter

    def log_pass(
        self,
        query_id: int,
        query: str,
        mode: int,
        pass_num: int,
        phi_cosine: float,
        phi_content: float,
        delta_phi: float,
        delta_phi_2: float,
        governor_decision: str,
        halt_reason: Optional[str] = None,
        passes_used: Optional[int] = None,
    ) -> None:
        row = {
            "query_id":         query_id,
            "query_preview":    query[:60],
            "mode":             mode,
            "pass_num":         pass_num,
            "phi_cosine":       round(phi_cosine,  6),
            "phi_content":      round(phi_content, 6),
            "delta_phi":        round(delta_phi,   6),
            "delta_phi_2":      round(delta_phi_2, 6),
            "governor_decision":governor_decision,
            "halt_reason":      halt_reason or "",
            "passes_used":      passes_used or "",
            "timestamp":        datetime.now().isoformat(),
        }
        _append_csv(self.csv_path, self.FIELDS, row)


# ── Checkpoint Weight Logger ──────────────────────────────────────────────────

class CheckpointLogger:
    """
    Log LoRA weight norms after each training checkpoint.

    Verifies that:
    - Left and right LoRA adapters are diverging (different norms by epoch 2+)
    - Shared layers remain unchanged (norm delta ≈ 0)
    - No gradient explosion (norm < 100 is healthy)

    CSV output: {output_dir}/checkpoint_norms.csv
    """

    FIELDS = [
        "step", "epoch", "pathway", "layer_idx",
        "lora_A_norm", "lora_B_norm", "delta_weight_norm",
        "timestamp",
    ]

    def __init__(self, output_dir: str = "./logs"):
        self.csv_path = os.path.join(output_dir, "checkpoint_norms.csv")
        _ensure_csv(self.csv_path, self.FIELDS)

    def log_checkpoint(
        self,
        model,
        step: int,
        epoch: float,
        pathway_name: str,
    ) -> None:
        """
        Scan all LoRA modules in model and log their weight norms.

        Compatible with PEFT PeftModel: looks for lora_A and lora_B attributes.
        """
        logged = 0
        for name, module in model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                # Determine layer index from name (e.g., "model.layers.20.self_attn")
                parts     = name.split(".")
                layer_idx = next((p for p in parts if p.isdigit()), "?")

                # Norm of LoRA_A matrix (down projection)
                lora_a_tensor = (
                    module.lora_A.default.weight
                    if hasattr(module.lora_A, "default")
                    else list(module.lora_A.values())[0].weight
                )
                lora_b_tensor = (
                    module.lora_B.default.weight
                    if hasattr(module.lora_B, "default")
                    else list(module.lora_B.values())[0].weight
                )

                a_norm = lora_a_tensor.float().norm().item()
                b_norm = lora_b_tensor.float().norm().item()

                # Delta weight = (lora_B @ lora_A) scaled by alpha/r
                try:
                    scaling = module.scaling.get("default", 1.0)
                    delta   = (lora_b_tensor.float() @ lora_a_tensor.float()) * scaling
                    d_norm  = delta.norm().item()
                except Exception:
                    d_norm = 0.0

                row = {
                    "step":               step,
                    "epoch":              round(epoch, 2),
                    "pathway":            pathway_name,
                    "layer_idx":          layer_idx,
                    "lora_A_norm":        round(a_norm, 6),
                    "lora_B_norm":        round(b_norm, 6),
                    "delta_weight_norm":  round(d_norm, 6),
                    "timestamp":          datetime.now().isoformat(),
                }
                _append_csv(self.csv_path, self.FIELDS, row)
                logged += 1

        if logged == 0:
            print(f"  WARNING: No LoRA modules found in model at step {step}.")
        else:
            print(f"  [checkpoint] step={step}  epoch={epoch:.1f}  "
                  f"pathway={pathway_name}  logged {logged} LoRA modules")


# ── Session summary ───────────────────────────────────────────────────────────

def write_session_summary(
    output_dir: str,
    phase: str,
    pathway: Optional[str],
    notes: str = "",
) -> None:
    """
    Write a human-readable session summary JSON.
    Call at the end of each training phase.
    """
    summary = {
        "phase":     phase,
        "pathway":   pathway,
        "timestamp": datetime.now().isoformat(),
        "notes":     notes,
        "csv_files": [
            f for f in os.listdir(output_dir) if f.endswith(".csv")
        ] if os.path.isdir(output_dir) else [],
    }
    path = os.path.join(output_dir, f"session_{phase}_{pathway or 'all'}.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Session summary: {path}")
