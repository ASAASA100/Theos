#!/usr/bin/env bash
# =============================================================================
# THEOS Native — Llama 3 Access Monitor & Auto-Download
# =============================================================================
#
# Polls HuggingFace every 30 minutes to check if Llama 3 8B Instruct access
# has been approved. When approved, downloads the model and starts the server.
#
# Usage:
#   chmod +x monitor_and_download.sh
#   HF_TOKEN=hf_xxx ./monitor_and_download.sh &
#   # or run in a tmux/screen session:
#   tmux new-session -d -s theos './monitor_and_download.sh 2>&1 | tee /root/THEOS/monitor.log'
#
# Patent: U.S. Application No. 18/919,771 (Provisional #63/831,738)
# Author: Frederick Davis Stalnecker
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID="meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_DIR="/root/models/Meta-Llama-3-8B-Instruct"
NATIVE_DIR="/root/THEOS/native_theos"
LOG_FILE="/root/THEOS/monitor.log"
PID_FILE="/root/THEOS/server.pid"
VENV_DIR="/root/THEOS/venv"

CHECK_INTERVAL=1800   # 30 minutes in seconds
MAX_CHECKS=96         # Stop after 96 checks (48 hours)

# HuggingFace token — set via environment or hardcode here
HF_TOKEN="${HF_TOKEN:-}"

if [[ -z "$HF_TOKEN" ]]; then
    echo "ERROR: HF_TOKEN environment variable not set."
    echo "Usage: HF_TOKEN=hf_xxx ./monitor_and_download.sh"
    exit 1
fi

# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

# ---------------------------------------------------------------------------
# Check if Llama 3 access is approved
# ---------------------------------------------------------------------------

check_access() {
    # Try to list model files — returns 403 if not approved, 200 if approved
    local http_code
    http_code=$(curl -s -o /dev/null -w "%{http_code}" \
        -H "Authorization: Bearer ${HF_TOKEN}" \
        "https://huggingface.co/api/models/${MODEL_ID}")

    if [[ "$http_code" == "200" ]]; then
        return 0  # Access granted
    else
        return 1  # Not yet approved (403, 401, etc.)
    fi
}

# ---------------------------------------------------------------------------
# Download the model
# ---------------------------------------------------------------------------

download_model() {
    log "Llama 3 access approved! Starting download..."
    log "Model will be saved to: ${MODEL_DIR}"

    mkdir -p "$MODEL_DIR"

    # Activate virtual environment
    if [[ -f "${VENV_DIR}/bin/activate" ]]; then
        source "${VENV_DIR}/bin/activate"
    fi

    # Use huggingface_hub to download
    python3 - <<PYEOF
import sys
from huggingface_hub import snapshot_download

print("Downloading ${MODEL_ID}...")
print("This may take 10-20 minutes depending on connection speed.")

try:
    snapshot_download(
        repo_id="${MODEL_ID}",
        local_dir="${MODEL_DIR}",
        token="${HF_TOKEN}",
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
    )
    print("Download complete.")
except Exception as e:
    print(f"Download failed: {e}")
    sys.exit(1)
PYEOF

    log "Model download complete: ${MODEL_DIR}"
}

# ---------------------------------------------------------------------------
# Install Python dependencies if not already installed
# ---------------------------------------------------------------------------

install_deps() {
    log "Installing Python dependencies..."

    # Create venv if it doesn't exist
    if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
        log "Creating Python virtual environment at ${VENV_DIR}..."
        python3 -m venv "$VENV_DIR"
    fi

    source "${VENV_DIR}/bin/activate"

    # Install requirements
    pip install --quiet --upgrade pip

    REQUIREMENTS_FILE="${NATIVE_DIR}/requirements.txt"
    if [[ -f "$REQUIREMENTS_FILE" ]]; then
        pip install --quiet -r "$REQUIREMENTS_FILE"
        log "Dependencies installed from requirements.txt"
    else
        # Install core dependencies directly
        pip install --quiet \
            torch \
            transformers \
            accelerate \
            bitsandbytes \
            peft \
            fastapi \
            uvicorn[standard] \
            anthropic \
            huggingface_hub

        log "Core dependencies installed."
    fi
}

# ---------------------------------------------------------------------------
# Start the THEOS server
# ---------------------------------------------------------------------------

start_server() {
    log "Starting THEOS Native server..."

    if [[ -f "${VENV_DIR}/bin/activate" ]]; then
        source "${VENV_DIR}/bin/activate"
    fi

    cd "$NATIVE_DIR"

    # Kill any existing server
    if [[ -f "$PID_FILE" ]]; then
        old_pid=$(cat "$PID_FILE")
        if kill -0 "$old_pid" 2>/dev/null; then
            log "Stopping existing server (PID $old_pid)..."
            kill "$old_pid"
            sleep 2
        fi
    fi

    # Start server in background
    nohup python3 server.py >> /root/THEOS/server.log 2>&1 &
    server_pid=$!
    echo "$server_pid" > "$PID_FILE"

    log "THEOS server started (PID $server_pid)"
    log "Server log: /root/THEOS/server.log"
    log "API available at http://104.131.169.165:8000"
    log "Health check: http://104.131.169.165:8000/health"
}

# ---------------------------------------------------------------------------
# Run training data generation (Phase 3)
# ---------------------------------------------------------------------------

generate_training_data() {
    log "Generating training data with Claude..."

    if [[ -f "${VENV_DIR}/bin/activate" ]]; then
        source "${VENV_DIR}/bin/activate"
    fi

    cd "$NATIVE_DIR"

    if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
        log "WARNING: ANTHROPIC_API_KEY not set — skipping training data generation."
        log "Set ANTHROPIC_API_KEY and run: python3 generate_training_data.py"
        return
    fi

    log "Starting training data generation (100+ questions, ~10 minutes)..."
    python3 generate_training_data.py
    log "Training data generation complete. See training_data/"
}

# ---------------------------------------------------------------------------
# Main: poll loop
# ---------------------------------------------------------------------------

main() {
    log "====================================================="
    log "THEOS Native Monitor Started"
    log "Checking for Llama 3 access every 30 minutes..."
    log "Max wait: 48 hours (${MAX_CHECKS} checks)"
    log "====================================================="

    # Check if model already downloaded
    if [[ -d "$MODEL_DIR" ]] && [[ -f "${MODEL_DIR}/config.json" ]]; then
        log "Model already downloaded at ${MODEL_DIR}"
        log "Skipping download. Starting server..."
        install_deps
        start_server
        generate_training_data
        log "THEOS Native is running."
        exit 0
    fi

    # Install deps now (before model download, so we're ready)
    install_deps

    # Poll loop
    checks=0
    while [[ $checks -lt $MAX_CHECKS ]]; do
        checks=$((checks + 1))
        log "Access check ${checks}/${MAX_CHECKS}..."

        if check_access; then
            log "ACCESS GRANTED by Meta/HuggingFace!"
            download_model
            start_server
            generate_training_data
            log "====================================================="
            log "THEOS Native is fully operational."
            log "Server: http://104.131.169.165:8000"
            log "Health: curl http://104.131.169.165:8000/health"
            log "====================================================="
            exit 0
        else
            log "Access still pending. Next check in 30 minutes."
            sleep $CHECK_INTERVAL
        fi
    done

    log "ERROR: Access check timed out after 48 hours."
    log "Please check your HuggingFace access request status at:"
    log "  https://huggingface.co/${MODEL_ID}"
    exit 1
}

# ---------------------------------------------------------------------------
# Immediate download option (skip polling, for when access is already granted)
# ---------------------------------------------------------------------------

if [[ "${1:-}" == "--download-now" ]]; then
    log "Forced immediate download (--download-now flag)"
    install_deps
    download_model
    start_server
    generate_training_data
    exit 0
fi

if [[ "${1:-}" == "--start-server" ]]; then
    log "Starting server only (--start-server flag)"
    install_deps
    start_server
    exit 0
fi

if [[ "${1:-}" == "--status" ]]; then
    if [[ -f "$PID_FILE" ]]; then
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            echo "THEOS server is running (PID $pid)"
        else
            echo "THEOS server is NOT running (stale PID $pid)"
        fi
    else
        echo "THEOS server is NOT running (no PID file)"
    fi

    if [[ -f "${MODEL_DIR}/config.json" ]]; then
        echo "Model: downloaded at ${MODEL_DIR}"
    else
        echo "Model: NOT downloaded yet"
    fi

    check_access && echo "HuggingFace access: GRANTED" || echo "HuggingFace access: PENDING"
    exit 0
fi

# Run main poll loop
main
