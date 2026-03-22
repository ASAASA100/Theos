#!/bin/bash
# =============================================================
# THEOS Native Architecture — RunPod Setup Script
# =============================================================
# Run this once after SSHing into a fresh RunPod pod.
#
# Requirements:
#   - RunPod pod with CUDA GPU (A100 40GB recommended)
#   - HF_TOKEN env var set (HuggingFace token for Llama 3 access)
#   - ANTHROPIC_API_KEY env var set (for training data generation)
#
# Usage:
#   export HF_TOKEN=your_token
#   export ANTHROPIC_API_KEY=your_key
#   bash runpod_setup.sh
# =============================================================

set -e  # Exit on any error

echo "============================================================"
echo "THEOS Native Architecture — RunPod Setup"
echo "Author: Frederick Davis Stalnecker"
echo "============================================================"

# --- GPU check ---
echo ""
echo "--- GPU ---"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# --- Validate required env vars ---
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN not set. Run: export HF_TOKEN=your_huggingface_token"
    exit 1
fi
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "ERROR: ANTHROPIC_API_KEY not set. Run: export ANTHROPIC_API_KEY=your_key"
    exit 1
fi

# --- Clone or update repo ---
echo ""
echo "--- Repository ---"
cd /workspace
if [ -d "THEOS" ]; then
    echo "Repo exists — pulling latest..."
    cd THEOS && git pull
else
    echo "Cloning THEOS repo..."
    gh repo clone Frederick-Stalnecker/THEOS || git clone https://github.com/Frederick-Stalnecker/THEOS.git
    cd THEOS
fi

# --- Install dependencies ---
echo ""
echo "--- Installing dependencies ---"
pip install -q -r native_theos/requirements.txt

# --- HuggingFace login ---
echo ""
echo "--- HuggingFace authentication ---"
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

# --- Test CUDA + torch ---
echo ""
echo "--- Verifying PyTorch + CUDA ---"
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# --- Set working directory ---
cd /workspace/THEOS/native_theos

echo ""
echo "============================================================"
echo "Setup complete. Next steps:"
echo ""
echo "  1. Generate training data (~3 USD, ~15 min):"
echo "     cd /workspace/THEOS/native_theos"
echo "     python generate_training_data.py"
echo ""
echo "  2. Train left pathway (constructive, ~1-2 hrs):"
echo "     python train.py --phase left"
echo ""
echo "  3. Train right pathway (adversarial, ~1-2 hrs):"
echo "     python train.py --phase right"
echo ""
echo "  4. Train governor (~30 min):"
echo "     python train.py --phase governor"
echo ""
echo "  5. Test inference:"
echo "     python inference.py --query 'Should terminally ill patients have unrestricted access to experimental treatments?'"
echo ""
echo "  Or train all phases in sequence (unattended):"
echo "     python train.py --phase all 2>&1 | tee training.log"
echo "============================================================"
