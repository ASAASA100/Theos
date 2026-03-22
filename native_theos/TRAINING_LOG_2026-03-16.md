# THEOS Native Architecture — Training Log
**Date:** 2026-03-16
**Session:** RunPod A40 (47GB VRAM), Phase A+B+C complete, bugs found and fixed

---

## Session Summary

This session completed a full training run (Phase A left, Phase B right+POS, Phase C governor)
and then discovered fundamental bugs in the architecture that invalidated the results.
All bugs were fixed in code. The next training run will produce correct results.

---

## Training Run 1 (Phase A+B+C) — Results and Problems

| Phase | Outcome |
|-------|---------|
| Phase A (left, 3 epochs, rank 16) | Completed. Loss decreased. |
| Phase B (right+POS, 3 epochs, rank 16) | Completed. Loss decreased. |
| Phase C (governor, 10 epochs) | 47% accuracy, loss 1.11→0.71 |

**Governor accuracy 47%** — slightly above chance (25% for 4-class). Suspicious.

---

## Bugs Found and Fixed

### Bug 1: CPU-Merge Approach Kills LoRA Hooks (CRITICAL)

**Old approach:** `merge_and_unload()` on CPU, then extract layers, transfer to GPU.

**Problem:** `merge_and_unload()` bakes the LoRA delta into the base weights (`W₀ + BA`)
and returns a plain `LlamaForCausalLM`. The LoRA forward hooks are gone. At inference,
both pathways run with the SAME merged weights for any given query. No divergence is
possible. This is why Φ ~0.009.

**Fix (architecture.py lines 409–459):** Multi-adapter approach:
- Wrap already-loaded `base_model` with PEFT: `PeftModel.from_pretrained(base_model, left_dir, adapter_name="left")`
- Load right adapter: `base_model.load_adapter(right_dir, adapter_name="right")`
- Call `set_adapter("left")` before left pathway, `set_adapter("right")` before right pathway
- Memory: ~16.2 GB total vs ~32+ GB for CPU merge (was causing OOM on A40)
- LoRA hooks stay live — pathways genuinely diverge

### Bug 2: Phi_JS Always Zero (CRITICAL)

**Old code:** `compute_phi_js(shared_attn, shared_attn)` — comparing shared attention to itself.
JS divergence of any distribution vs itself = 0 by definition.

**Fix (architecture.py line 701):**
- Capture `attn_L` from left pathway's last layer (with `capture_last_attn=True`)
- Capture `attn_R` from right pathway's last layer (with `capture_last_attn=True`)
- Compute `compute_phi_js(attn_L, attn_R)` — LEFT vs RIGHT, genuinely different after LoRA

### Bug 3: Governor Trained on Word-Overlap Targets (CRITICAL)

**Root cause:** `generate_training_data.py` uses `estimate_text_similarity()` (Jaccard word
overlap) as `phi_proxy`. This gives phi_proxy ~0.6–0.8 for adversarial text pairs.
`governor_data.jsonl` stores these values as `target` labels.

**Problem:** Governor was trained on word-overlap targets (~0.6–0.8 range) but receives
real hidden-state Φ (~0.009 for untrained pathways). The distributions are completely
unrelated. The Governor never learned anything meaningful.

**Fix (train.py lines 479–491):** Always derive target from live `phi_comp`:
- Pass 1: always `target = 0` (continue) — explore before deciding
- Pass 2+ with `phi_comp < eps_converge (0.05)`: `target = 1` (halt_converged)
- Pass 2+ with `phi_comp > eps_partial (0.5)`: `target = 2` (halt_disagree)
- Pass 2+ middle range: `target = 0` (continue)

The `target` field in JSONL is now ignored.

### Bug 4: load_trained_pathways OOM Risk

**Old approach:** Loaded TWO additional full base models (~16 GB each) on GPU to merge
left and right adapters separately. Total GPU usage ~48 GB on a 47 GB A40.

**Fix (train.py load_trained_pathways):** Same multi-adapter approach as architecture.py.
Wrap `native_model.base_model` with both adapters in-place. No second base model loaded.

---

## Architecture Understanding: Why Φ ~0.009 Before Fix

With CPU-merge (Bug 1), both pathways ran with identical merged weights:
- Left path ran with `W₀ + B_L A_L` baked in
- Right path ran with `W₀ + B_R A_R` baked in (different merge, but...)
- BUT: they were assigned to the SAME `self.right_layers = self.left_layers` in the no-adapter
  branch — so BOTH used the left-merged weights!

Even with correct separate layers, at rank 16 + 3 epochs + 97 training examples, the LoRA
delta is small. Higher rank + more epochs + more data is needed for meaningful Φ.

---

## What "Φ ~0.8 Yesterday" Actually Was

The Φ ~0.8 printed during `generate_training_data.py` is `phi_proxy` from
`estimate_text_similarity()` — Jaccard word overlap between constructive and adversarial
text responses. This is NOT hidden-state divergence. It was misleading.

Target real Φ after fixes: **0.10–0.30** (enough to run multi-pass wringer, not so high
the pathways polarize and become incoherent).

---

## Next Training Run Parameters

| Parameter | Old | New |
|-----------|-----|-----|
| LoRA rank | 16 | 32 |
| Epochs (pathways) | 3 | 10 |
| Training examples | 97 | 120 (full set) |
| Adapter mode | CPU merge (hooks dead) | Multi-adapter (hooks live) |
| Governor targets | Word-overlap proxy | Live phi_comp |
| Phi_JS | shared_attn vs shared_attn (= 0) | attn_L vs attn_R (correct) |

**Command:**
```bash
cd /workspace/THEOS/native_theos
python generate_training_data.py          # ~15 min, ~$3 API cost
python train.py --phase all --epochs 10 --lora-rank 32 2>&1 | tee training_run2.log
```

---

## Files Changed This Session

| File | Change |
|------|--------|
| `native_theos/architecture.py` | Multi-adapter LoRA loading, set_adapter() calls, fix Phi_JS |
| `native_theos/train.py` | Fix governor targets (live phi), fix load_trained_pathways (multi-adapter) |

---

## Patent Note

The multi-adapter approach (single base model + both LoRA adapters via PEFT's multi-adapter
API, switched via `set_adapter()`) is the correct implementation of:
- **Claim 21** (composite Φ now computed from genuinely different pathway attentions)
- **Claim 23** (cross-attention between pathways requires real diverged hidden states)

The previous CPU-merge approach was architecturally incorrect — it could not produce
genuinely dialectical reasoning because both pathways were computing identically.

---

*Author: Frederick Davis Stalnecker*
*Filed: 2026-03-16*

---

## Training Run 2 Results (after bug fixes)

### Phase A — Left pathway (constructive)
- Rank 32, 10 epochs, 97 examples
- Loss: 2.469 → 1.917 (avg final)
- Runtime: 3m 4s on A40

### Phase B — Right pathway (adversarial + POS loss)
- Rank 32, 10 epochs, 97 examples  
- POS loss starts high (~11) → decreases as diversity increases
- Runtime: ~3 min on A40

### Phase C — Governor MLP
- 10 epochs, 194 governor examples
- **Accuracy: 75.3%** (vs 47% in Run 1 — major improvement)
- Loss: 0.9469 → 0.4636
- Live phi_comp values: 0.028–0.090 (real, not word-overlap)
- phi_js still 0 (SDPA attention doesn't support output_attentions → fixed with eager)

---

## First Successful Wringer Run — 2026-03-16

Query: "Should terminally ill patients have unrestricted access to experimental treatments?"

| Metric | Value |
|--------|-------|
| Φ_composite | 0.1318 |
| Φ_cosine | 0.0742 |
| Φ_content | 0.3212 |
| Passes | 3 (Governor: "continue" all 3) |
| Halt reason | budget_exhaustion_partial |
| Output type | blend (w_L=0.37, w_R=0.63) |
| Confidence | 86.82% |

**This is the first successful multi-pass wringer run in THEOS native architecture.**

The two pathways genuinely diverged:
- Left: builds affirmative case (moral imperative + clinical integrity balance)
- Right: challenges the framing itself ("unrestricted" vs "reasonable" access)

Φ_content = 0.32 is meaningful (pathways predict different tokens).
Φ_composite = 0.1318 is above eps_converge (0.05) — wringer correctly continued all 3 passes.

### Additional bugs fixed post-training

4. **Cross-attention dtype mismatch (Pass 2 crash):**
   - cross_attn module was float32, hidden states bfloat16
   - Fix: `.to(_dtype)` for cross_attn only (governor stays float32)

5. **attn_implementation='eager' added:**
   - SDPA attention silently ignores output_attentions=True → phi_js = 0
   - Fix: load_kwargs includes attn_implementation='eager'
   - phi_js will now be non-zero after next training run uses eager attention during training

---

## Status: WRINGER IS RUNNING

The native THEOS V2 architecture is proven working end-to-end:
✓ Multi-adapter LoRA (hooks live)  
✓ Y-fork dual pathway generation
✓ Composite Phi (cosine + content both > 0)
✓ Governor trained on real phi (75% accuracy)
✓ Multi-pass wringer (3 passes on this query)
✓ Cross-attention initialized (not yet trained — will strengthen on next run)
✓ Synthesis output (blend w_L/w_R weighted by phi_cosine)

phi_js = 0 remains an issue (eager attn fix pushed, will work after next load).

Next: Generate more training data (need ANTHROPIC_API_KEY), retrain with eager attention enabled.
