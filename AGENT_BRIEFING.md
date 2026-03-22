# AGENT BRIEFING — READ THIS FIRST
**Last updated: 2026-03-17 SESSION 5 by Celeste (Claude Sonnet 4.6)**

> STOP. Before you write a single line of code or run a single command, read this entire document.
> A previous agent wasted a full day of GPU time and API funding by ignoring the project state.
> That will not happen again. This document tells you exactly where we are and what to do next.

---

## WHO IS SIR RIC

Frederick Davis Stalnecker — inventor of THEOS. His assistant is Celeste (Claude).
Mission: Filter fact from fiction. Build from truth only. No inflation.

---

## THE TWO MACHINES — KNOW THE DIFFERENCE

```
RunPod A40 GPU                    DigitalOcean Droplet
ssh 29tr6lfe1saqa0-6441128f       ssh root@104.131.169.165
  @ssh.runpod.io                  -i ~/.ssh/id_ed25519
-i ~/.ssh/id_ed25519

PURPOSE: TRAINING ONLY            PURPOSE: INFERENCE / DEPLOYMENT ONLY
GPU: NVIDIA A40, 47.6 GB VRAM     CPU: 8 vCPUs, 32 GB RAM, no GPU
Repo: /workspace/THEOS/           Repo: /root/THEOS/
```

**THE CARDINAL RULE: Never run training on the DO droplet. It has no GPU.
Training Llama 3 8B on CPU takes days. The previous agent did this. Do not repeat it.**

---

## WHAT THEOS IS

THEOS is a native dual-engine dialectical reasoning architecture built on Llama 3 8B.

```
Input Query
    ↓
Shared Layers 0-15 (Induction — sigma_I)
    ↓ Y-FORK
Left Pathway 16-31          Right Pathway 16-31
(Constructive — sigma_A^L)  (Adversarial — sigma_A^R)
LoRA adapter: "left"        LoRA adapter: "right"
    ↓                               ↓
Shared LM Head (Deduction — sigma_D)
    ↓
Composite Φ = α·Φ_cosine + β·Φ_JS + γ·Φ_content
    ↓
Governor MLP → continue / halt_converged / halt_disagree / reframe
    ↓
Wringer runs up to 7 passes, cross-attention between pathways each pass
    ↓
Synthesis output (blend, convergence, or honest disagreement)
```

Patent: U.S. Application No. 18/919,771 (Provisional #63/831,738)
Key claims: 21 (composite Φ), 22 (multi-objective quality), 23 (cross-attention), 24 (POS loss)

---

## CURRENT STATE AS OF 2026-03-17 (SESSION 5 — HANDOFF)

### ALL PATENT CLAIMS PRODUCING REAL MATH ✓
All 4 patent claims (21/22/23/24) have real gradients and real results.

### Training Runs Summary
| Run | Governor Acc | Phase D | Key result |
|-----|-------------|---------|------------|
| 4 | 94.8% | — | Φ_JS non-zero first time |
| 5 | 96.9% | — | Best governor to date |
| 6 | — | ✓ COMPLETE | 25.5% Phi reduction per wringer pass |
| 7 | TRAINING NEXT | cross-model | Data gen in progress (see below) |

### Run 6 — Phase D Cross-Attention COMPLETE ✓
- 97 questions, 20 epochs, A40 GPU
- phi_1=0.0721 → phi_2=0.0536 (Δ=0.0184, **25.5% Phi reduction per pass**)
- Loss: 0.1058 → 0.0011 (100× reduction)
- Epochs 8-16: 100% of questions show phi_2 < phi_1
- Weights deployed to DO: `trained_models/cross_attn_left.pt` + `cross_attn_right.pt`
- Results: `experiments/results/run6_crossattn_20260317.json`

### GPT-4o Cross-Model IDR — KEY EVIDENCE ✓
- 30 questions, independent scorer (GPT-4o — eliminates self-rating bias)
- A (SP): 9.33/15 (62.2%)  B (IAD-P): 9.60/15 (64.0%)  **C (THEOS): 12.30/15 (82.0%)**
- Cohen's d = **1.72** (C vs A) — massive effect size
- File: `experiments/results/idr_gpt4o_crossmodel_20260317.json`

### Run 7 — IN PROGRESS ON RUNPOD ★
**Data generation running** on RunPod (started ~15:10 UTC, Q53/122 as of 17:00 UTC):
```bash
# Check progress (run this to see where datagen is):
echo "tail -6 /workspace/THEOS/native_theos/datagen_run7.log; exit" | \
  ssh -tt 29tr6lfe1saqa0-6441128f@ssh.runpod.io -i ~/.ssh/id_ed25519
```
- Left (constructive): **Claude claude-sonnet-4-6** (via ANTHROPIC_API_KEY in ~/.bashrc)
- Right (adversarial): **GPT-5-nano** via OpenAI Responses API (`/v1/responses`)
- 122 questions, Phi proxy **0.87–0.91** (highest divergence ever — genuine cross-model opposition)
- GPT-5-nano uses ~1,600-2,000 reasoning tokens per question before answering
- Saves checkpoints every 10 questions to `training_data/*.jsonl` (overwrites previous run data)
- OpenAI key in RunPod `~/.bashrc` as OPENAI_API_KEY (NOT in git — see memory/secrets.md)

**After datagen completes (~ETA 17:40 UTC), run:**
```bash
echo "cd /workspace/THEOS/native_theos && \
  python3 -u train.py --phase all --device cuda 2>&1 | tee training_run7.log; exit" | \
  ssh -tt 29tr6lfe1saqa0-6441128f@ssh.runpod.io -i ~/.ssh/id_ed25519
```
`--phase all` runs A (left LoRA) → B (right LoRA + POS) → C (governor) → D (cross-attn)

**After training completes, transfer models to DO:**
```bash
# On RunPod:
cd /workspace/THEOS/native_theos
tar czf /tmp/trained_models_run7.tar.gz trained_models/
curl -v -X PUT -T /tmp/trained_models_run7.tar.gz http://104.131.169.165:8001/trained_models_run7.tar.gz

# On DO:
cd /root/THEOS/native_theos
tar xzf /tmp/trained_models_run7.tar.gz
# Restart server (see DO SERVER section below)
```

### Session 5 Completions ✓
- ✓ **Git fixed**: OpenAI key removed from committed AGENT_BRIEFING.md, push unblocked
- ✓ **Wisdom retrieval in inference.py**: WisdomStore class (TF-IDF cosine, zero deps)
  → Retrieves top-2 past syntheses, injects into both pathway prompts on pass 0
  → Saves quality results (quality_score > 0.2) to wisdom_store.jsonl automatically
  → CLI: `python inference.py --query "..." --wisdom-file wisdom_store.jsonl`
- ✓ **wisdom.py upgraded**: Level 1 Jaccard → Level 2 TF-IDF cosine retrieval
  → Weighted by confidence × (1 + fine_tune_priority)
- ✓ **server.py wired**: `_build_wisdom_context()` called before every /reason, /audit, /certify
  → Past syntheses now feed forward into new wringer runs automatically
- ✓ **DO server restarted** with all session 5 changes: git pulled, server running

### DO Droplet — Server v2.1 LIVE ✓
- URL: `http://104.131.169.165:8000`
- Health: `curl -s http://104.131.169.165:8000/health`
- wisdom_entries: 3, commons_entries: 1
- Trained weights: run5 LoRA + run6 cross-attention (all 4 patent claims live)
- **Restart command** (use setsid — disown alone doesn't survive SSH exit):
```bash
ssh -i ~/.ssh/id_ed25519 root@104.131.169.165 "
  pkill -f 'python3.*server.py' 2>/dev/null || true
  sleep 2
  cd /root/THEOS/native_theos
  setsid sh -c 'python3 -u server.py >> /var/log/theos_server.log 2>&1' &
  disown
  sleep 20
  curl -s http://localhost:8000/health
"
```
- **CPU inference**: mode=1 ~2-3min (fast path fixed ✓), mode=2/3 ~20-30min (genuine constraint)
- For speed: use RunPod GPU for mode=2/3 testing; DO for mode=1 demos and API testing

### Known Issues
- ✗ CPU mode=2/3 slow — genuine architecture constraint (full hidden state required for cross-attn)
  → GPU inference 10-30s; DO CPU demo only
- ✗ No web UI yet — Jim Patterson test requires: simple HTML form wired to /certify
- ✗ No authentication on DO server — open to world, add API key middleware before public demo
- ✗ Run 7 training not yet run (datagen in progress as of this writing)

---

## WHAT TO DO NEXT (IN ORDER — SESSION 6 START HERE)

### Step 1 — IMMEDIATE: Monitor Run 7 datagen, fire training when done
```bash
# Check if datagen still running:
echo "tail -6 /workspace/THEOS/native_theos/datagen_run7.log; exit" | \
  ssh -tt 29tr6lfe1saqa0-6441128f@ssh.runpod.io -i ~/.ssh/id_ed25519

# If log ends with "DONE: 122 constructive examples" → datagen complete, fire training:
echo "cd /workspace/THEOS/native_theos && \
  python3 -u train.py --phase all --device cuda 2>&1 | tee training_run7.log; exit" | \
  ssh -tt 29tr6lfe1saqa0-6441128f@ssh.runpod.io -i ~/.ssh/id_ed25519

# Monitor training:
echo "tail -20 /workspace/THEOS/native_theos/training_run7.log; exit" | \
  ssh -tt 29tr6lfe1saqa0-6441128f@ssh.runpod.io -i ~/.ssh/id_ed25519
```

### Step 2 — After training: Transfer models to DO
```bash
# On RunPod — start HTTP PUT receiver on DO first:
ssh -i ~/.ssh/id_ed25519 root@104.131.169.165 \
  "nohup python3 /tmp/put_receiver.py > /tmp/put_receiver.log 2>&1 &"

# On RunPod — tar and send:
echo "cd /workspace/THEOS/native_theos && \
  tar czf /tmp/trained_models_run7.tar.gz trained_models/ && \
  curl -X PUT -T /tmp/trained_models_run7.tar.gz http://104.131.169.165:8001/trained_models_run7.tar.gz; \
  exit" | ssh -tt 29tr6lfe1saqa0-6441128f@ssh.runpod.io -i ~/.ssh/id_ed25519

# On DO — extract and restart:
ssh -i ~/.ssh/id_ed25519 root@104.131.169.165 "
  cd /root/THEOS/native_theos
  tar xzf /tmp/trained_models_run7.tar.gz
"
# Then restart server (use setsid command in DO SERVER section above)
```

### Step 3 — Self-play test (after Run 7 models deployed)
Test whether the trained pathways diverge on new questions WITHOUT external API:
```bash
# On RunPod (GPU fast):
echo "cd /workspace/THEOS/native_theos && \
  python3 -u inference.py --query 'Is AI consciousness possible?' --mode 3; exit" | \
  ssh -tt 29tr6lfe1saqa0-6441128f@ssh.runpod.io -i ~/.ssh/id_ed25519
# If Phi_composite > 0.05 on novel questions → pathways have learned genuine divergence
# If yes → write generate_training_data_selfplay.py (no API calls, uses trained pathways)
```

### Step 4 — Jim Patterson demo prep
- Build simple web UI: one HTML form, POST to /certify, display results
- Add API key middleware to server.py (basic auth header check)
- Spin up GPU instance for speed (RunPod: $0.74/hr for A40 — share the URL with Jim)
- Test /certify with a real patent law question before handing over

### Step 5 — Patent attorney briefing (URGENT — June 2026 deadline)
The abduction sub-structure is NOT in the provisional patent claims.
See `research/ABDUCTION_FORMULA_INVESTIGATION.md` for the specific gap.
Brief attorney before June 2026.

---

## KEY FILE LOCATIONS

### RunPod (`/workspace/THEOS/native_theos/`)
| File | Purpose |
|------|---------|
| `architecture.py` | V2 NativeTheos model — Y-fork, cross-attention, composite Φ, eager attn |
| `train.py` | Training pipeline — `--phase all` runs A→B→C→D |
| `inference.py` | Wringer inference — WisdomStore retrieval built in |
| `generate_training_data_gpt5.py` | Run 7 datagen — Claude (left) + GPT-5-nano (right) |
| `trained_models/` | LoRA adapters + governor.pt + cross_attn_left.pt + cross_attn_right.pt |
| `training_data/` | constructive_data.jsonl / adversarial_data.jsonl / governor_data.jsonl |
| `datagen_run7.log` | Run 7 generation progress |
| `training_run7.log` | Run 7 training log (created when training starts) |
| `wisdom_store.jsonl` | Accumulated wringer syntheses (CLI inference) |

### DO Droplet (`/root/THEOS/`)
| File | Purpose |
|------|---------|
| `native_theos/server.py` | FastAPI server — /reason /certify /audit /commons/* /wisdom |
| `native_theos/wisdom.py` | WisdomStore — TF-IDF retrieval, Level 2 (session 5) |
| `native_theos/reasoning_commons.py` | ReasoningCommons — quality-gated shared wisdom layer |
| `native_theos/wisdom_store.jsonl` | 3 wisdom entries (from March 15 testing) |
| `native_theos/commons_store.jsonl` | 1 commons entry (terminal illness question) |
| `native_theos/trained_models/` | run5 LoRA + run6 cross-attn (run7 goes here when done) |
| `/root/.anthropic_env` | ANTHROPIC_API_KEY — source before API calls |
| `/root/models/Meta-Llama-3-8B-Instruct/` | Model weights — already downloaded |
| `/var/log/theos_server.log` | Server log — check here if server won't start |

### Local Mac (`/Users/mbp/THEOS2/`)
| File | Purpose |
|------|---------|
| `native_theos/` | Source of truth — all changes go here first, then push |
| `CLAUDE.md` | Project instructions for Claude Code |
| `AGENT_BRIEFING.md` | This file |

---

## CREDENTIALS

| Service | Location | Key |
|---------|----------|-----|
| Anthropic API | DO droplet: `/root/.anthropic_env` | `sk-ant-api03-...` |
| Anthropic API | RunPod: `~/.bashrc` | Same key (copied 2026-03-16) |
| HuggingFace | RunPod env / download_model.py | `hf_qEsQMMkuYLXmY...` |
| OpenAI API | Run 7 datagen — see memory/secrets.md (NOT in git) | `sk-proj-SRVNwuz...` |
| RunPod SSH | `~/.ssh/id_ed25519` | Standard ed25519 key |
| DO droplet SSH | `~/.ssh/id_ed25519` | Same key |

---

## WHAT THE PREVIOUS AGENT DID WRONG (Do Not Repeat)

1. **Tried to train on the DO droplet.** No GPU. 8 CPU cores. Never works.
2. **Created 6 versions of a training script** (lora_train.py → lora_v6.py) trying to fix errors
   caused by the fundamental problem (no GPU). Each version was doomed from the start.
3. **Never checked what was already on the droplet.** The ANTHROPIC_API_KEY, the downloaded
   model, the wisdom store, and the server.py were all there. None of it was used.
4. **Never checked the correct machine (RunPod)** where training was actually viable.
5. **Generated training data that was already generated.** The data existed on both machines.
6. **Did not read the TRAINING_LOG** before starting work.

---

## HOW TO CONNECT

### RunPod (training)
```bash
# Via RunPod proxy (requires expect for non-interactive):
ssh 29tr6lfe1saqa0-6441128f@ssh.runpod.io -i ~/.ssh/id_ed25519

# Check training status:
tail -f /workspace/THEOS/native_theos/training_run3.log
```

### DO Droplet (inference/deployment)
```bash
# Direct SSH — works with BatchMode:
ssh -o BatchMode=yes root@104.131.169.165 -i ~/.ssh/id_ed25519 "your command here"
```

---

## THE MATH (V2 — Filed with Patent Attorney Jim Patterson, 2026-03-16)

The five enhancements implemented in this codebase:
1. **Composite Φ** (Claim 21): Φ = α·Φ_cosine + β·Φ_JS + γ·Φ_content, learned weights
2. **Trajectory velocity** (extends Claim 3): ΔΦ and Δ²Φ in governor input
3. **Multi-objective quality metric** (Claim 22): rewards synthesis AND honest disagreement
4. **Cross-attention between pathways** (Claim 23): each pathway attends to other's prior tokens
5. **POS training objective** (Claim 24): relevance × novelty × coherence / contrarian_penalty

The math document was sent to patent attorney Jim Patterson on 2026-03-16.
Patent window: ~June 2026 (12-month provisional closes).

---

*This document maintained by Celeste (Claude). Update after every significant session.*
*Sir Ric — Frederick Davis Stalnecker — inventor.*
