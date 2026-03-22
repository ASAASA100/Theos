#!/bin/bash
# auto_train_run4.sh — Autonomous training run 4
# Waits for data generation to finish, then runs full training.
# Run with: nohup bash auto_train_run4.sh > auto_train_run4.log 2>&1 &

set -euo pipefail
cd /workspace/THEOS/native_theos

echo "=== AUTO TRAIN RUN 4 — $(date) ==="
echo "Waiting for data generation to complete..."

# Wait for generate_training_data.py to finish
while pgrep -f "generate_training_data.py" > /dev/null; do
    echo "$(date) — data gen still running..."
    sleep 30
done

echo "$(date) — data generation complete."

# Count training examples
CONSTR=$(wc -l < training_data/constructive_data.jsonl 2>/dev/null || echo 0)
ADV=$(wc -l < training_data/adversarial_data.jsonl 2>/dev/null || echo 0)
GOV=$(wc -l < training_data/governor_data.jsonl 2>/dev/null || echo 0)
echo "Training data: ${CONSTR} constructive, ${ADV} adversarial, ${GOV} governor examples"

# Pull latest code (phi_JS fix)
echo "$(date) — pulling latest code..."
cd /workspace/THEOS && git pull origin main && cd native_theos
echo "$(date) — code updated."

# Source API keys
source ~/.bashrc

echo ""
echo "=== STARTING TRAINING RUN 4 — $(date) ==="
echo "Phase: all (A + B + C)"
echo "LoRA rank: 32, epochs: 10"
echo "phi_JS fix: hidden-state JS divergence (works with transformers 5.3.0)"
echo ""

# Run training
python -u train.py --phase all --epochs 10 --lora-rank 32 2>&1 | tee training_run4.log

echo ""
echo "=== TRAINING RUN 4 COMPLETE — $(date) ==="
echo ""

# Quick inference test to verify phi_JS is now non-zero
echo "=== RUNNING INFERENCE VERIFICATION ==="
python -u inference.py \
    --query "Should terminally ill patients have unrestricted access to experimental treatments?" \
    --max-passes 3 \
    2>&1 | tee inference_run4_verify.log

echo ""
echo "=== DONE — $(date) ==="
echo "Check training_run4.log and inference_run4_verify.log for results."
