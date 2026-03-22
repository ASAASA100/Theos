"""
THEOS Native Inference Engine — Version 2
==========================================

Implements the full 3-mode wringer loop with V2 enhancements:

    Claim 21 — Composite Phi: alpha*Phi_cosine + beta*Phi_JS + gamma*Phi_content
    Claim 22 — Multi-objective quality metric in trace
    Claim 23 — Full hidden state passed between passes for cross-attention
    Claim 24 — POS score logged per pass

Adaptive Modes (Frederick Stalnecker, March 15 2026):
    Mode 1 - Reflective:   Single engine, 2 IAD cycles  (~500-1000 tokens)
    Mode 2 - Dialectical:  Dual engine, 1-3 passes      (~1500-3000 tokens)
    Mode 3 - Full Wringer: Dual engine, up to 7 passes  (~3000-7000 tokens)

Governor decisions:
    continue        - Keep the wringer running
    halt_converged  - Engines agree, output constructive answer
    halt_disagree   - Irresolvable tension, expose both sides
    reframe         - Hidden premise detected, surface it first

Patent: U.S. Application No. 18/919,771 (Provisional #63/831,738)
Author: Frederick Davis Stalnecker
"""

import json
import math
import os
import re
import torch
import argparse
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple
from architecture import NativeTheos, NativeTheosConfig


# ============================================================
# WISDOM STORE — Accumulated synthesis across wringer sessions
# ============================================================

class WisdomStore:
    """
    Persistent store of past THEOS syntheses for cross-session learning.

    Retrieval uses TF-IDF cosine similarity — pure Python, zero dependencies.
    When a past synthesis is found with similarity >= threshold, it is injected
    into both pathway system prompts before the wringer runs.

    This implements the "accumulated wisdom" property: each THEOS instance
    learns from every previous wringer run stored in the file.
    """

    def __init__(self, path: str, threshold: float = 0.4):
        self.path = path
        self.threshold = threshold
        self._entries: List[dict] = []
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        self._entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    def save(self, query: str, output: str, output_type: str,
             confidence: float, phi_final: float, passes: int) -> None:
        entry = {
            "query":       query,
            "output":      output[:1000],   # cap stored synthesis at 1k chars
            "output_type": output_type,
            "confidence":  round(confidence, 4),
            "phi_final":   round(phi_final, 4),
            "passes":      passes,
            "timestamp":   datetime.now().isoformat(),
        }
        self._entries.append(entry)
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def _tfidf_vec(self, tokens: List[str], idf: dict) -> dict:
        if not tokens:
            return {}
        tf = Counter(tokens)
        return {w: (c / len(tokens)) * idf[w] for w, c in tf.items() if w in idf}

    def retrieve(self, query: str, top_k: int = 2) -> List[dict]:
        """Return up to top_k entries whose query is similar to this query."""
        if not self._entries:
            return []

        # Build IDF over all stored queries + current query
        all_docs = [self._tokenize(e["query"]) for e in self._entries]
        all_docs.append(self._tokenize(query))
        N = len(all_docs)
        df: Counter = Counter()
        for doc in all_docs:
            for w in set(doc):
                df[w] += 1
        idf = {w: math.log((N + 1) / (df[w] + 1)) + 1.0 for w in df}

        q_tokens = self._tokenize(query)
        q_vec    = self._tfidf_vec(q_tokens, idf)
        q_norm   = math.sqrt(sum(v ** 2 for v in q_vec.values())) + 1e-10

        scored = []
        for entry in self._entries:
            e_tokens = self._tokenize(entry["query"])
            e_vec    = self._tfidf_vec(e_tokens, idf)
            e_norm   = math.sqrt(sum(v ** 2 for v in e_vec.values())) + 1e-10
            dot      = sum(q_vec.get(w, 0.0) * e_vec.get(w, 0.0) for w in q_vec)
            sim      = dot / (q_norm * e_norm)
            if sim >= self.threshold:
                scored.append((sim, entry))

        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored[:top_k]]


COMPLEX_KEYWORDS = {
    "should", "ought", "ethical", "moral", "tradeoff", "trade-off",
    "dilemma", "conflict", "risk", "versus", "vs", "balance",
    "rights", "justice", "fair", "harm", "benefit", "policy",
    "legal", "illegal", "medical", "diagnosis", "treatment",
    "allocate", "prioritize", "decide", "judge", "weigh",
}

SIMPLE_KEYWORDS = {
    "what is", "who is", "when", "where", "how many", "define",
    "capital of", "list", "name", "spell", "calculate", "convert",
}


def detect_mode(query: str) -> int:
    q_lower = query.lower()
    for kw in SIMPLE_KEYWORDS:
        if kw in q_lower:
            return 1
    complex_count = sum(1 for kw in COMPLEX_KEYWORDS if kw in q_lower)
    if complex_count >= 3:
        return 3
    elif complex_count >= 1:
        return 2
    return 1


@dataclass
class NativeWringerPassTrace:
    """Record of one native wringer pass — V2 includes all three Phi channels."""
    wringer_pass:     int
    phi_cosine:       float
    phi_js:           float
    phi_content:      float
    phi_composite:    float
    delta_phi:        float
    delta_phi_2:      float
    governor_decision: str
    left_output:      str
    right_output:     str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class NativeTheosOutput:
    """
    Output from native THEOS V2 reasoning.

    V2 additions:
        phi_trajectory_composite  — composite Phi trajectory
        phi_trajectory_cosine     — cosine channel
        phi_trajectory_js         — attention channel
        phi_trajectory_content    — output distribution channel
        quality_score             — multi-objective Q (Claim 22)
    """
    output:                    str
    output_type:               str
    confidence:                float
    contradiction:             float
    wringer_passes_used:       int
    halt_reason:               str
    mode_used:                 int
    phi_trajectory_composite:  List[float]
    phi_trajectory_cosine:     List[float]
    phi_trajectory_js:         List[float]
    phi_trajectory_content:    List[float]
    quality_score:             float
    trace:                     List[NativeWringerPassTrace]
    query:                     str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


SYSTEM_PROMPT_LEFT = (
    "You are reasoning constructively. Your role is to build the strongest, "
    "most thoughtful affirmative answer. Develop the best case for the most "
    "defensible position. Be clear and substantive."
)

SYSTEM_PROMPT_RIGHT = (
    "You are reasoning adversarially. Your role is to challenge assumptions, "
    "identify weaknesses, and steelman the strongest objection. Find what the "
    "constructive view misses. Be incisive and specific."
)


def _build_pathway_input(
    model: NativeTheos,
    query: str,
    pathway: str,
    wisdom_context: Optional[str] = None,
) -> torch.Tensor:
    base_system = SYSTEM_PROMPT_LEFT if pathway == "left" else SYSTEM_PROMPT_RIGHT
    if wisdom_context:
        system = f"{base_system}\n\nRELEVANT PRIOR SYNTHESIS:\n{wisdom_context}"
    else:
        system = base_system
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": query},
    ]
    try:
        token_list = model.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True,
        )
        return torch.tensor([token_list], dtype=torch.long).to(model.config.device)
    except Exception:
        text = f"[SYSTEM: {system}]\n\nUser: {query}\n\nAssistant:"
        enc = model.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        return enc.input_ids.to(model.config.device)


# ============================================================
# V2 CLAIM 22 — MULTI-OBJECTIVE QUALITY METRIC
# ============================================================

def compute_quality_score(
    phi_trajectory: List[float],
    halt_decision: str,
) -> float:
    """
    Multi-objective quality metric Q (Claim 22).

    Four trace archetypes:
        Genuine synthesis:    Phi high -> low, converged        (highest Q)
        Honest disagreement:  Phi stable-high, irresolvable     (valuable)
        Trivial agreement:    Phi low throughout                (low Q)
        Failed reasoning:     Phi oscillates, no resolution     (penalized)

    Q = (w1*Q_synth + w2*Q_disagree + w3*Q_reframe) / (1 + lambda*P_osc)

    Patent: Claim 22 — U.S. Application No. 18/919,771
    """
    n = len(phi_trajectory)
    if n == 0:
        return 0.0
    if n == 1:
        return phi_trajectory[0] * 0.5   # single pass, partial credit

    # Trajectory velocity
    deltas = [phi_trajectory[i] - phi_trajectory[i-1] for i in range(1, n)]

    # Smoothness: mean absolute velocity (low = stable, high = oscillating)
    sigma_traj = sum(abs(d) for d in deltas) / len(deltas)

    phi_first = phi_trajectory[0]
    phi_last  = phi_trajectory[-1]

    # Oscillation penalty: fraction of sign-flips in velocity
    if len(deltas) > 1:
        sign_flips = sum(
            1 for i in range(1, len(deltas))
            if (deltas[i] * deltas[i-1]) < 0
        )
        p_osc = sign_flips / (len(deltas) - 1)
    else:
        p_osc = 0.0

    # Component scores
    if halt_decision == "halt_converged":
        # Synthesis: started high, ended low, converged
        q_synth    = phi_first * (1.0 - phi_last) * (1.0 / (1.0 + sigma_traj))
        q_disagree = 0.0
        q_reframe  = 0.0
    elif halt_decision == "halt_disagree":
        # Honest disagreement: stable-high Phi, correctly identified irresolvable
        q_synth    = 0.0
        q_disagree = phi_first * (1.0 / (1.0 + sigma_traj))
        q_reframe  = 0.0
    elif halt_decision == "reframe":
        # Reframe: caught false premise early (fewer passes = higher quality)
        q_synth    = 0.0
        q_disagree = 0.0
        q_reframe  = 1.0 / n
    else:
        # Budget exhaustion
        q_synth    = phi_first * (1.0 - phi_last) * 0.5 * (1.0 / (1.0 + sigma_traj))
        q_disagree = 0.0
        q_reframe  = 0.0

    # Weights (equal by default; could be learned in future)
    w1, w2, w3 = 0.5, 0.35, 0.15
    lam = 2.0   # oscillation penalty strength

    q = (w1 * q_synth + w2 * q_disagree + w3 * q_reframe) / (1.0 + lam * p_osc)
    return round(float(q), 4)


# ============================================================
# WRINGER LOOP
# ============================================================

def run_native_wringer(
    model: NativeTheos,
    query: str,
    mode: Optional[int] = None,
    verbose: bool = True,
    wisdom_store: Optional[WisdomStore] = None,
    wisdom_context: Optional[str] = None,
) -> NativeTheosOutput:
    """
    Run the complete native THEOS V2 wringer loop.

    V2 key differences from V1:
        - past_H_L / past_H_R are FULL hidden state tensors [batch, seq, hidden_dim]
          (not mean-pooled vectors) — fed to cross-attention in next pass
        - All three Phi channels tracked separately and as composite
        - Quality score computed at end of trace (Claim 22)
    """
    config = model.config

    if mode is None:
        mode = detect_mode(query)

    max_passes = {
        1: config.mode1_max_passes,
        2: config.mode2_max_passes,
        3: config.mode3_max_passes,
    }.get(mode, config.mode3_max_passes)

    if verbose:
        print(f"\n{'='*60}")
        print(f"THEOS Native Wringer V2")
        print(f"Query: {query}")
        print(f"Mode: {mode} | Max passes: {max_passes}")
        print(f"{'='*60}\n")

    trace                     = []
    phi_traj_composite        = []
    phi_traj_cosine           = []
    phi_traj_js               = []
    phi_traj_content          = []

    # V2: full hidden state tensors for cross-attention (Claim 23)
    past_H_L: Optional[torch.Tensor] = None   # [batch, seq_len, hidden_dim]
    past_H_R: Optional[torch.Tensor] = None

    # Wisdom retrieval — seed both pathways with past relevant synthesis
    # wisdom_context can be pre-built by the caller (e.g., server.py) or
    # built here from a WisdomStore (CLI path).
    if wisdom_context is None and wisdom_store is not None:
        past = wisdom_store.retrieve(query, top_k=2)
        if past:
            snippets = []
            for p in past:
                snippets.append(
                    f"Past query: {p['query']}\n"
                    f"Synthesis ({p['output_type']}, confidence={p['confidence']:.0%}):\n"
                    f"{p['output'][:400]}"
                )
            wisdom_context = "\n\n---\n\n".join(snippets)
            if verbose:
                print(f"  [Wisdom] {len(past)} prior synthesis(es) retrieved.")
                for p in past:
                    print(f"    • {p['query'][:70]}...")

    prev_phi_composite: float = 0.0
    prev_delta_phi:     float = 0.0

    final_left_text  = ""
    final_right_text = ""
    final_phi        = 0.0

    for wringer_pass in range(max_passes):
        if verbose:
            print(f"--- Wringer Pass {wringer_pass + 1} ---")

        # Inject wisdom context only on pass 0 (seeds initial reasoning)
        wctx = wisdom_context if wringer_pass == 0 else None
        ids_L = _build_pathway_input(model, query, "left",  wisdom_context=wctx)
        ids_R = _build_pathway_input(model, query, "right", wisdom_context=wctx)

        # === MODE 1: Single engine only ===
        if mode == 1:
            if verbose:
                print("  Mode 1 — generating constructive response (KV-cached fast path)...")
            # generate_from_pathway uses base_model.generate() with KV cache —
            # safe for mode=1 because no Phi or cross-attention is needed here.
            # ~100x faster than generate_through_pathway on CPU.
            left_text, _ = model.generate_from_pathway(
                ids_L, pathway="left",
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
            )
            phi_traj_composite.append(0.0)
            phi_traj_cosine.append(0.0)
            phi_traj_js.append(0.0)
            phi_traj_content.append(0.0)
            trace.append(NativeWringerPassTrace(
                wringer_pass=wringer_pass, phi_cosine=0.0, phi_js=0.0,
                phi_content=0.0, phi_composite=0.0, delta_phi=0.0, delta_phi_2=0.0,
                governor_decision="halt_converged",
                left_output=left_text[:300], right_output=left_text[:300],
            ))
            q = compute_quality_score(phi_traj_composite, "halt_converged")
            return NativeTheosOutput(
                output=left_text, output_type="convergence",
                confidence=1.0, contradiction=0.0,
                wringer_passes_used=1, halt_reason="mode1_single_engine",
                mode_used=mode,
                phi_trajectory_composite=phi_traj_composite,
                phi_trajectory_cosine=phi_traj_cosine,
                phi_trajectory_js=phi_traj_js,
                phi_trajectory_content=phi_traj_content,
                quality_score=q, trace=trace, query=query,
            )

        # === MODES 2 & 3: Dual engine ===
        if verbose:
            print("  Generating constructive response (left pathway)...")
        left_text, h_L_pooled, H_L_full = model.generate_through_pathway(
            ids_L, pathway="left",
            past_H_L=past_H_L, past_H_R=past_H_R,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
        )

        if verbose:
            print("  Generating adversarial response (right pathway)...")
        right_text, h_R_pooled, H_R_full = model.generate_through_pathway(
            ids_R, pathway="right",
            past_H_L=past_H_L, past_H_R=past_H_R,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
        )

        final_left_text  = left_text
        final_right_text = right_text

        # Run a probe forward pass to get all three Phi channels cleanly
        with torch.inference_mode():
            probe = model.forward_single_pass(
                ids_L[:, :min(256, ids_L.shape[1])],
                past_H_L=past_H_L,
                past_H_R=past_H_R,
            )

        phi_cosine_val  = probe["phi_cosine"].mean().item()
        phi_js_val      = probe["phi_js"].mean().item()
        phi_content_val = probe["phi_content"].mean().item()
        phi_comp_val    = probe["phi"].mean().item()

        final_phi = phi_comp_val

        phi_traj_composite.append(round(phi_comp_val, 4))
        phi_traj_cosine.append(round(phi_cosine_val, 4))
        phi_traj_js.append(round(phi_js_val, 4))
        phi_traj_content.append(round(phi_content_val, 4))

        # Trajectory velocity and acceleration (Enhancement 2)
        delta_phi   = phi_comp_val - prev_phi_composite
        delta_phi_2 = delta_phi - prev_delta_phi

        # Governor decision
        phi_cosine_t  = torch.tensor([[phi_cosine_val]],  dtype=torch.float32).to(config.device)
        phi_js_t      = torch.tensor([[phi_js_val]],      dtype=torch.float32).to(config.device)
        phi_content_t = torch.tensor([[phi_content_val]], dtype=torch.float32).to(config.device)
        delta_phi_t   = torch.tensor([[delta_phi]],       dtype=torch.float32).to(config.device)
        delta_phi_2_t = torch.tensor([[delta_phi_2]],     dtype=torch.float32).to(config.device)
        pass_num_t    = torch.tensor(
            [[wringer_pass / max_passes]], dtype=torch.float32
        ).to(config.device)

        h_L = h_L_pooled if h_L_pooled.dim() == 2 else h_L_pooled.mean(dim=1)
        h_R = h_R_pooled if h_R_pooled.dim() == 2 else h_R_pooled.mean(dim=1)

        governor_logits = model.governor(
            h_L.float(), h_R.float(),
            phi_cosine_t, phi_js_t, phi_content_t,
            delta_phi_t, delta_phi_2_t, pass_num_t,
        )
        governor_idx      = torch.softmax(governor_logits, dim=-1).argmax(dim=-1).item()
        governor_decisions = ["continue", "halt_converged", "halt_disagree", "reframe"]
        governor_decision  = governor_decisions[governor_idx]

        # Rule-based override for untrained Governor
        if phi_comp_val < config.eps_converge:
            governor_decision = "halt_converged"
        elif phi_comp_val > config.eps_partial and wringer_pass >= 2:
            governor_decision = "halt_disagree"

        if verbose:
            w = model.governor.channel_weights.detach()
            print(f"  Φ_cos={phi_cosine_val:.4f}  Φ_JS={phi_js_val:.4f}  Φ_content={phi_content_val:.4f}")
            print(f"  Φ_composite={phi_comp_val:.4f}  (α={w[0]:.2f} β={w[1]:.2f} γ={w[2]:.2f})")
            print(f"  ΔΦ={delta_phi:+.4f}  Δ²Φ={delta_phi_2:+.4f}")
            print(f"  Governor: {governor_decision}")
            print(f"  Left:  {left_text[:100].strip()}...")
            print(f"  Right: {right_text[:100].strip()}...")

        trace.append(NativeWringerPassTrace(
            wringer_pass=wringer_pass,
            phi_cosine=phi_cosine_val, phi_js=phi_js_val,
            phi_content=phi_content_val, phi_composite=phi_comp_val,
            delta_phi=delta_phi, delta_phi_2=delta_phi_2,
            governor_decision=governor_decision,
            left_output=left_text[:300], right_output=right_text[:300],
        ))

        prev_delta_phi       = delta_phi
        prev_phi_composite   = phi_comp_val

        # === HALT CONDITIONS ===
        if governor_decision == "halt_converged":
            q = compute_quality_score(phi_traj_composite, "halt_converged")
            confidence = max(0.0, min(1.0, 1.0 - (phi_comp_val / max(config.eps_converge, 1e-6))))
            result = NativeTheosOutput(
                output=left_text, output_type="convergence",
                confidence=min(confidence, 1.0), contradiction=phi_comp_val,
                wringer_passes_used=wringer_pass + 1, halt_reason="convergence",
                mode_used=mode,
                phi_trajectory_composite=phi_traj_composite,
                phi_trajectory_cosine=phi_traj_cosine,
                phi_trajectory_js=phi_traj_js,
                phi_trajectory_content=phi_traj_content,
                quality_score=q, trace=trace, query=query,
            )
            if wisdom_store is not None and q > 0.2:
                wisdom_store.save(query, left_text, "convergence",
                                  float(min(confidence, 1.0)), phi_comp_val, wringer_pass + 1)
            return result

        elif governor_decision == "halt_disagree":
            q = compute_quality_score(phi_traj_composite, "halt_disagree")
            output = (
                f"CONSTRUCTIVE ENGINE:\n{left_text}\n\n"
                f"ADVERSARIAL ENGINE:\n{right_text}\n\n"
                f"COMPOSITE Φ: {phi_comp_val:.4f}  "
                f"(Φ_cos={phi_cosine_val:.4f}  Φ_JS={phi_js_val:.4f}  Φ_content={phi_content_val:.4f})\n"
                f"The engines could not reach agreement. "
                f"Both perspectives are presented for your consideration."
            )
            return NativeTheosOutput(
                output=output, output_type="disagreement",
                confidence=0.5, contradiction=phi_comp_val,
                wringer_passes_used=wringer_pass + 1, halt_reason="irreducible_uncertainty",
                mode_used=mode,
                phi_trajectory_composite=phi_traj_composite,
                phi_trajectory_cosine=phi_traj_cosine,
                phi_trajectory_js=phi_traj_js,
                phi_trajectory_content=phi_traj_content,
                quality_score=q, trace=trace, query=query,
            )

        elif governor_decision == "reframe":
            q = compute_quality_score(phi_traj_composite, "reframe")
            output = (
                f"REFRAME DETECTED:\n"
                f"The question contains a premise both engines struggled with.\n\n"
                f"CONSTRUCTIVE perspective on the premise: {left_text}\n\n"
                f"ADVERSARIAL perspective on the premise: {right_text}\n\n"
                f"Consider reframing your question before proceeding."
            )
            return NativeTheosOutput(
                output=output, output_type="reframe",
                confidence=0.6, contradiction=phi_comp_val,
                wringer_passes_used=wringer_pass + 1, halt_reason="reframe",
                mode_used=mode,
                phi_trajectory_composite=phi_traj_composite,
                phi_trajectory_cosine=phi_traj_cosine,
                phi_trajectory_js=phi_traj_js,
                phi_trajectory_content=phi_traj_content,
                quality_score=q, trace=trace, query=query,
            )

        # === CONTINUE: Store FULL hidden states for next pass cross-attention ===
        past_H_L = H_L_full.detach()   # [batch, seq_len, hidden_dim]
        past_H_R = H_R_full.detach()

    # === BUDGET EXHAUSTED ===
    if verbose:
        print("Budget exhausted — applying output rule")

    if final_phi < config.eps_converge:
        output_type = "convergence"
        output      = final_left_text
        confidence  = max(0.0, 1.0 - (final_phi / config.eps_converge))
        halt_reason = "budget_exhaustion_converged"
    elif final_phi < config.eps_partial:
        output_type = "blend"
        w_L = (1 - final_phi / config.eps_partial) / 2
        w_R = (1 + final_phi / config.eps_partial) / 2
        output = (
            f"BLENDED OUTPUT (w_L={w_L:.2f}, w_R={w_R:.2f}):\n\n"
            f"CONSTRUCTIVE ({w_L:.0%}):\n{final_left_text}\n\n"
            f"ADVERSARIAL  ({w_R:.0%}):\n{final_right_text}"
        )
        confidence  = max(0.0, 1.0 - (final_phi / config.eps_partial) * 0.5)
        halt_reason = "budget_exhaustion_partial"
    else:
        output_type = "disagreement"
        output = (
            f"CONSTRUCTIVE ENGINE:\n{final_left_text}\n\n"
            f"ADVERSARIAL ENGINE:\n{final_right_text}\n\n"
            f"COMPOSITE Φ: {final_phi:.4f}"
        )
        confidence  = 0.5
        halt_reason = "budget_exhaustion"

    q = compute_quality_score(phi_traj_composite, halt_reason)
    final_confidence = min(max(confidence, 0.0), 1.0)
    result = NativeTheosOutput(
        output=output, output_type=output_type,
        confidence=final_confidence, contradiction=final_phi,
        wringer_passes_used=max_passes, halt_reason=halt_reason,
        mode_used=mode,
        phi_trajectory_composite=phi_traj_composite,
        phi_trajectory_cosine=phi_traj_cosine,
        phi_trajectory_js=phi_traj_js,
        phi_trajectory_content=phi_traj_content,
        quality_score=q, trace=trace, query=query,
    )
    if wisdom_store is not None and q > 0.2 and output_type in ("convergence", "blend"):
        wisdom_store.save(query, output, output_type,
                          final_confidence, final_phi, max_passes)
    return result


def main():
    parser = argparse.ArgumentParser(description="THEOS Native Inference V2")
    parser.add_argument("--query",        type=str, required=True)
    parser.add_argument("--mode",         type=int, default=None, choices=[1, 2, 3])
    parser.add_argument("--max-passes",   type=int, default=7)
    parser.add_argument("--no-verbose",   action="store_true")
    parser.add_argument("--wisdom-file",  type=str,
                        default=os.path.join(os.path.dirname(__file__), "wisdom_store.jsonl"),
                        help="Path to wisdom store JSONL file (default: wisdom_store.jsonl)")
    parser.add_argument("--no-wisdom",    action="store_true",
                        help="Disable wisdom retrieval and saving for this run")
    args = parser.parse_args()

    import torch as _torch
    _device = "cuda" if _torch.cuda.is_available() else "cpu"
    config = NativeTheosConfig(
        max_wringer_passes=args.max_passes,
        mode1_max_passes=args.max_passes,
        mode2_max_passes=args.max_passes,
        mode3_max_passes=args.max_passes,
        device=_device,
    )
    model  = NativeTheos(config)

    wisdom_store = None if args.no_wisdom else WisdomStore(args.wisdom_file)
    if wisdom_store and not args.no_verbose:
        print(f"Wisdom store: {args.wisdom_file} ({len(wisdom_store._entries)} entries)")

    result = run_native_wringer(
        model, args.query, mode=args.mode,
        verbose=not args.no_verbose,
        wisdom_store=wisdom_store,
    )

    print(f"\n{'='*60}")
    print("THEOS NATIVE V2 RESULT")
    print(f"{'='*60}")
    print(f"Output Type:      {result.output_type}")
    print(f"Mode Used:        {result.mode_used}")
    print(f"Confidence:       {result.confidence:.2%}")
    print(f"Contradiction:    {result.contradiction:.4f}")
    print(f"Passes Used:      {result.wringer_passes_used}")
    print(f"Quality Score:    {result.quality_score:.4f}  (Claim 22)")
    print(f"Halt Reason:      {result.halt_reason}")
    print(f"Φ composite:      {result.phi_trajectory_composite}")
    print(f"Φ cosine:         {result.phi_trajectory_cosine}")
    print(f"Φ JS:             {result.phi_trajectory_js}")
    print(f"Φ content:        {result.phi_trajectory_content}")
    print(f"\n--- OUTPUT ---\n{result.output}")
    print(f"\n--- TRACE ---")
    for t in result.trace:
        print(f"  Pass {t.wringer_pass + 1}: Φ={t.phi_composite:.4f} → {t.governor_decision}")


if __name__ == "__main__":
    main()
