#!/bin/bash
# auto_train_run5.sh — Autonomous training run 5
# Waits for data generation to finish, then runs full training.
# Run with: nohup bash auto_train_run5.sh > auto_train_run5.log 2>&1 &
#
# Fixed vs run4: removed set -euo pipefail (source ~/.bashrc fails with PS1 unbound in non-interactive shell)

set -e
cd /workspace/THEOS/native_theos

echo "=== AUTO TRAIN RUN 5 — $(date) ==="
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

# Pull latest code
echo "$(date) — pulling latest code..."
cd /workspace/THEOS && git pull origin main && cd native_theos
echo "$(date) — code updated."

# Load environment (best-effort — non-interactive bash may not have PS1)
set +e
export ANTHROPIC_API_KEY=$(grep 'ANTHROPIC_API_KEY' ~/.bashrc | grep -o '"sk-ant[^"]*"' | tr -d '"' 2>/dev/null)
export HF_TOKEN=$(grep 'HF_TOKEN' ~/.bashrc | grep -o '"hf_[^"]*"' | tr -d '"' 2>/dev/null)
set -e

echo ""
echo "=== STARTING TRAINING RUN 5 — $(date) ==="
echo "Phase: all (A + B + C)"
echo "LoRA rank: 32, epochs: 10"
echo "Fresh training data: ${CONSTR} constructive / ${ADV} adversarial / ${GOV} governor"
echo ""

# Run training
python -u train.py --phase all --epochs 10 --lora-rank 32 2>&1 | tee training_run5.log

echo ""
echo "=== TRAINING RUN 5 COMPLETE — $(date) ==="
echo ""

# Run inference verification
echo "=== RUNNING INFERENCE VERIFICATION ==="
python -u inference.py \
    --query "Should terminally ill patients have unrestricted access to experimental treatments?" \
    --max-passes 3 \
    2>&1 | tee inference_run5_verify.log

echo ""
python -u inference.py \
    --query "Is consciousness a prerequisite for moral consideration?" \
    --max-passes 7 \
    2>&1 | tee inference_run5_consciousness.log

echo ""
echo "=== DONE — $(date) ==="
echo "Check training_run5.log and inference_run5_verify.log for results."
