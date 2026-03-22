"""
THEOS Native Architecture: Y-Forked Dual-Engine Transformer — Version 2
========================================================================

Implements the THEOS dual-engine dialectical reasoning architecture
natively in PyTorch on top of Llama 3 8B.

Architecture:
    Shared Layers (0 to split_point-1)  -> sigma_I  (Induction)
    Left Pathway  (split_point to N)    -> sigma_A^L (Constructive Abduction)
    Right Pathway (split_point to N)    -> sigma_A^R (Adversarial Abduction)
    Shared LM Head                      -> sigma_D  (Deduction)
    Governor MLP                        -> Halt/Continue decision

Adaptive Modes (Frederick Stalnecker, March 15 2026):
    Mode 1 - Reflective:   Single engine, 2 IAD cycles  (~500-1000 tokens)
    Mode 2 - Dialectical:  Dual engine, 1-3 passes      (~1500-3000 tokens)
    Mode 3 - Full Wringer: Dual engine, up to 7 passes  (~3000-7000 tokens)

The Governor's four decisions:
    continue        - Keep the wringer running
    halt_converged  - Engines agree, output constructive answer
    halt_disagree   - Irresolvable tension, expose both sides
    reframe         - Hidden premise detected, surface it first

V2 Enhancements (Frederick Stalnecker, March 16 2026):
    Claim 21 — Composite divergence: Phi = alpha*Phi_cosine + beta*Phi_JS + gamma*Phi_content
               Three pre-decoded divergence channels with learned weighting.
               Phi_JS measures attention pattern divergence (JS across heads).
    Claim 22 — Multi-objective quality metric: rewards synthesis, honest disagreement,
               and early reframe. Penalizes oscillation. Distinguishes irresolvability
               from reasoning failure.
    Claim 23 — Pathway cross-attention: each pathway attends to specific token
               positions in the other's prior hidden state, not a mean-pooled summary.
               This is genuine dialectical response — responding to what was said,
               not to an averaged impression of it.
    Claim 24 — Productive Opposition Score (POS) training objective: decomposes
               adversarial quality into relevance x novelty x coherence / contrarian_penalty.
               Prevents the two failure modes: contrarianism and parroting.

Patent: U.S. Application No. 18/919,771 (Provisional #63/831,738)
Author: Frederick Davis Stalnecker
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from datetime import datetime


@dataclass
class NativeTheosConfig:
    """Configuration for the native THEOS architecture."""

    # Base model
    base_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    split_layer: int = 16           # Where the Y-fork splits (0-15 shared, 16-31 diverge)

    # Governor thresholds (from TheosConfig in theos_core.py)
    max_wringer_passes: int = 7
    eps_converge: float = 0.05      # Below this Phi -> convergence
    eps_partial: float = 0.5        # Above this Phi -> disagreement
    rho_min: float = 0.4            # Min info gain ratio
    entropy_min: float = 0.15       # Min entropy for irreducible check
    delta_min: float = 0.4          # Min Phi for irreducible check

    # Adaptive mode thresholds
    mode1_max_passes: int = 2       # Reflective: single engine, 2 IAD cycles
    mode2_max_passes: int = 3       # Dialectical: dual engine, up to 3 passes
    mode3_max_passes: int = 7       # Full wringer: up to 7 passes

    # Generation
    max_new_tokens: int = 150
    temperature: float = 0.7

    # Memory
    load_in_4bit: bool = False   # auto-detected by VRAM; False = use bfloat16 direct
    device: str = "cpu"

    # V2: Cross-attention
    cross_attn_heads: int = 8       # Number of heads in pathway cross-attention module


# ============================================================
# V2 CLAIM 23 — PATHWAY CROSS-ATTENTION MODULE
# ============================================================

class CrossAttentionModule(nn.Module):
    """
    Pathway cross-attention: each pathway attends to specific token positions
    in the other pathway's prior hidden state sequence.

    This replaces the V1 mean-pooled Momentary Past prefix. Instead of each
    pathway seeing a single averaged vector representing the other's prior
    state, each pathway can now attend to the specific tokens where the other
    pathway found weakness or emphasis.

    This is how actual argument works: you respond to what was said, not to
    an averaged impression of it.

    CrossAttn(H_source, H_query) =
        softmax((H_query W_Q) (H_source W_K)^T / sqrt(d_k)) (H_source W_V)

    At pass n >= 2:
        H_tilde_L^(n) = H_shared + CrossAttn(H_R^(n-1), H_shared)
        H_tilde_R^(n) = H_shared + CrossAttn(H_L^(n-1), H_shared)

    The query is the current shared hidden state (what this pathway is about to
    reason over). The source is the OTHER pathway's previous hidden state (what
    it should respond to).

    Patent: Claim 23 — U.S. Application No. 18/919,771
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Separate projection matrices — not shared with self-attention
        self.W_Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_K = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_V = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_O = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Layer norm for residual stability
        self.norm = nn.LayerNorm(hidden_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize near-zero so cross-attention starts as identity (no-op at pass 1)."""
        nn.init.normal_(self.W_Q.weight, std=0.01)
        nn.init.normal_(self.W_K.weight, std=0.01)
        nn.init.normal_(self.W_V.weight, std=0.01)
        nn.init.zeros_(self.W_O.weight)   # zero init: residual starts as pure H_shared

    def forward(
        self,
        H_source: torch.Tensor,   # [batch, src_len, hidden_dim] — other pathway's prior state
        H_query: torch.Tensor,    # [batch, qry_len, hidden_dim] — current shared state
    ) -> torch.Tensor:
        """
        Returns cross-attended context [batch, qry_len, hidden_dim].
        Add this to H_shared to form H_tilde before the pathway layers.
        """
        B, S, D = H_source.shape
        _, Q, _ = H_query.shape

        # Project queries, keys, values
        q = self.W_Q(H_query).view(B, Q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_K(H_source).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_V(H_source).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale   # [B, heads, Q, S]
        attn_weights = F.softmax(scores, dim=-1)

        out = torch.matmul(attn_weights, v)                     # [B, heads, Q, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, Q, D)    # [B, Q, hidden_dim]
        out = self.W_O(out)

        return self.norm(out)


# ============================================================
# V2 CLAIM 21 — GOVERNOR MLP WITH COMPOSITE DIVERGENCE
# ============================================================

class GovernorMLP(nn.Module):
    """
    The THEOS Governor as a learned neural module — Version 2.

    V2 additions:
      - Receives composite Phi (alpha*Phi_cosine + beta*Phi_JS + gamma*Phi_content)
        as a single scalar, alongside the three raw components separately.
      - Learnable alpha, beta, gamma (softmax-normalized) — the system discovers
        the relative diagnostic value of each divergence signal.
      - Input is now 2d+7: H_L + H_R + Phi_cosine + Phi_JS + Phi_content +
        delta_phi + delta_phi_2 + pass_num

    Input (2d+7):
        H_L_pooled   (hidden_dim) — left pathway mean-pooled hidden state
        H_R_pooled   (hidden_dim) — right pathway mean-pooled hidden state
        phi_cosine   (1)          — 1 - cos(H_L, H_R), in [0, 2]
        phi_js       (1)          — JS(p_L || p_R) on attention patterns, in [0, 1]
        phi_content  (1)          — JS(P_L || P_R) on output distributions, in [0, 1]
        delta_phi    (1)          — Phi^(n) - Phi^(n-1), trajectory velocity
        delta_phi_2  (1)          — delta_Phi^(n) - delta_Phi^(n-1), acceleration
        pass_num     (1)          — n / N_max, normalized pass index

    Output: [P(continue), P(halt_converged), P(halt_disagree), P(reframe)]

    Learnable composite:
        Phi = alpha * Phi_cosine + beta * Phi_JS + gamma * Phi_content
        alpha, beta, gamma = softmax([log_alpha, log_beta, log_gamma])

    Patent: Claims 21, 22 — U.S. Application No. 18/919,771
    """

    def __init__(self, hidden_dim: int = 4096):
        super().__init__()

        # 2d + 6 scalars: phi_cosine, phi_js, phi_content, delta_phi, delta_phi_2, pass_num
        # (one extra vs V1's 2d+5 — added phi_js)
        input_dim = hidden_dim * 2 + 6

        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 4),      # continue, halt_converged, halt_disagree, reframe
        )

        # V2 Claim 21: learnable divergence channel weights (log-space for stability)
        # Initialized equal (log(1/3) each); softmax gives alpha=beta=gamma=1/3 at init
        self.log_weights = nn.Parameter(
            torch.full((3,), math.log(1.0 / 3.0))
        )

    @property
    def channel_weights(self) -> torch.Tensor:
        """Returns (alpha, beta, gamma) normalized via softmax. Sum = 1."""
        return F.softmax(self.log_weights, dim=0)

    def composite_phi(
        self,
        phi_cosine: torch.Tensor,   # [batch, 1]
        phi_js:     torch.Tensor,   # [batch, 1]
        phi_content: torch.Tensor,  # [batch, 1]
    ) -> torch.Tensor:
        """
        Phi = alpha * Phi_cosine + beta * Phi_JS + gamma * Phi_content

        alpha, beta, gamma are learned. Different domains discover different
        optimal weightings through training.

        Patent: Claim 21.
        """
        w = self.channel_weights          # [3]
        return w[0] * phi_cosine + w[1] * phi_js + w[2] * phi_content

    def forward(
        self,
        h_left:      torch.Tensor,   # [batch, hidden_dim]
        h_right:     torch.Tensor,   # [batch, hidden_dim]
        phi_cosine:  torch.Tensor,   # [batch, 1]
        phi_js:      torch.Tensor,   # [batch, 1]
        phi_content: torch.Tensor,   # [batch, 1]
        delta_phi:   torch.Tensor,   # [batch, 1]
        delta_phi_2: torch.Tensor,   # [batch, 1]
        pass_num:    torch.Tensor,   # [batch, 1]
    ) -> torch.Tensor:
        """Returns logits for [continue, halt_converged, halt_disagree, reframe]."""
        x = torch.cat(
            [h_left, h_right, phi_cosine, phi_js, phi_content,
             delta_phi, delta_phi_2, pass_num],
            dim=-1,
        )
        return self.network(x)


# ============================================================
# V2 CLAIM 21 — PHI_JS: ATTENTION PATTERN DIVERGENCE
# ============================================================

def compute_phi_js(
    attn_L: torch.Tensor,   # [batch, heads, seq_len, seq_len] or list of layer attns
    attn_R: torch.Tensor,   # same shape
) -> torch.Tensor:
    """
    Phi_JS = (1/H) sum_h JS(p^h_L || p^h_R)

    where p^h_L[i] = (1/T) sum_j A^h_L[i,j]  (row-marginal attention distribution)

    Jensen-Shannon divergence is bounded [0, log(2)] and symmetric (unlike KL).
    Normalized to [0, 1] by dividing by log(2).

    This measures whether the pathways are ATTENDING to different parts of the
    question — a pre-decoded signal captured before any token is chosen.
    Pure attention weights. No output produced.

    Patent: Claim 21 — U.S. Application No. 18/919,771
    """
    # attn: [batch, heads, seq_len, seq_len]
    B, H, T, _ = attn_L.shape

    # Row-marginal: average over key dimension -> [batch, heads, seq_len]
    p_L = attn_L.mean(dim=-1).float()   # [B, H, T]
    p_R = attn_R.mean(dim=-1).float()   # [B, H, T]

    # Normalize rows to valid probability distributions (attention rows already sum ~1)
    p_L = p_L / (p_L.sum(dim=-1, keepdim=True) + 1e-10)
    p_R = p_R / (p_R.sum(dim=-1, keepdim=True) + 1e-10)

    m = 0.5 * (p_L + p_R)  # mixture distribution

    # KL(p || m) = sum p * log(p / m)
    def kl(p, q):
        return (p * ((p + 1e-10) / (q + 1e-10)).log()).sum(dim=-1)  # [B, H]

    js_per_head = 0.5 * kl(p_L, m) + 0.5 * kl(p_R, m)   # [B, H]
    phi_js = js_per_head.mean(dim=-1)                      # [B], average over heads

    # Normalize to [0, 1]
    phi_js = (phi_js / math.log(2)).clamp(0.0, 1.0)
    return phi_js.unsqueeze(-1)   # [B, 1]


# ============================================================
# NATIVE THEOS MODEL
# ============================================================

class NativeTheos(nn.Module):
    """
    The Native THEOS Dual-Engine Transformer — Version 2.

    V2 changes:
      - CrossAttentionModule replaces mean-pooled prefix for Momentary Past
      - GovernorMLP now receives all three Phi channels separately
      - Phi_JS captured from final shared layer attention weights
      - Learned alpha/beta/gamma composite divergence weights

    The forward pass implements one wringer pass:
        1. Shared layers process input (Induction sigma_I)
           Capture attention weights from final shared layer for Phi_JS.
        2. Left pathway: CrossAttn(H_R^(n-1), H_shared) + pathway layers
        3. Right pathway: CrossAttn(H_L^(n-1), H_shared) + pathway layers
        4. Compute composite Phi, Governor decides halt/continue/reframe

    The 'momentary past' (V2):
        Instead of prepending mean-pooled vectors as prefix tokens, the
        cross-attention module lets each pathway attend to specific token
        positions in the other pathway's prior hidden state. The left pathway
        can now respond to the SPECIFIC tokens where the right pathway found
        weakness — not just a blurred average of its argument.

    Patent: U.S. Application No. 18/919,771 (Provisional #63/831,738)
    Author: Frederick Davis Stalnecker
    """

    def __init__(self, config: NativeTheosConfig):
        super().__init__()
        self.config = config

        print(f"Loading base model: {config.base_model_name}")
        # Auto-select precision by VRAM. 4-bit QLoRA requires PyTorch >= 2.5
        # (set_submodule). For A40/A100 (>= 24GB) use bfloat16 directly.
        vram_gb = (
            torch.cuda.get_device_properties(0).total_memory / 1e9
            if torch.cuda.is_available() else 0
        )
        use_4bit = config.load_in_4bit and torch.cuda.is_available() and vram_gb < 24.0

        if use_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs = dict(
                quantization_config=BitsAndBytesConfig(load_in_4bit=True),
                device_map=config.device,
                low_cpu_mem_usage=True,
                attn_implementation="eager",   # required for output_attentions=True (Phi_JS)
            )
        else:
            load_kwargs = dict(
                dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
                device_map=config.device,
                low_cpu_mem_usage=True,
                attn_implementation="eager",   # required for output_attentions=True (Phi_JS)
            )

        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name,
            **load_kwargs,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        llama_model = base_model.model

        # === SHARED COMPONENTS (Induction sigma_I) ===
        self.embed_tokens   = llama_model.embed_tokens
        self.rotary_emb     = llama_model.rotary_emb
        self.shared_layers  = nn.ModuleList(
            [llama_model.layers[i] for i in range(config.split_layer)]
        )
        self.shared_norm    = copy.deepcopy(llama_model.norm)

        # === LEFT PATHWAY (Constructive Abduction sigma_A^L) ===
        self.left_layers    = nn.ModuleList(
            [llama_model.layers[i] for i in range(config.split_layer, len(llama_model.layers))]
        )
        self.left_norm      = copy.deepcopy(llama_model.norm)

        # === RIGHT PATHWAY (Adversarial Abduction sigma_A^R) ===
        # Shares base weights at init; LoRA fine-tuning diverges them.
        self.right_layers   = self.left_layers
        self.right_norm     = self.left_norm

        # Save lm_head before potential PEFT wrapping changes base_model type
        _lm_head_ref = base_model.lm_head

        # === LOAD TRAINED LoRA ADAPTERS (multi-adapter mode — LoRA hooks stay live) ===
        # Single base model + both LoRA adapters via PEFT multi-adapter API.
        # set_adapter("left" / "right") switches which LoRA is active before each
        # pathway forward pass — no second base-model load, no CPU merge, no OOM.
        # Memory: ~16.2 GB total (base bfloat16 + two tiny adapter dicts) vs
        # ~32+ GB for the old CPU-merge approach.
        import os as _os
        _models_dir    = _os.path.join(_os.path.dirname(__file__), "trained_models")
        _left_dir      = _os.path.join(_models_dir, "left_pathway")
        _right_dir     = _os.path.join(_models_dir, "right_pathway")
        _left_has      = _os.path.exists(_os.path.join(_left_dir,  "adapter_config.json"))
        _right_has     = _os.path.exists(_os.path.join(_right_dir, "adapter_config.json"))

        self.has_lora_adapters  = False
        self._has_left_adapter  = False
        self._has_right_adapter = False

        if _left_has or _right_has:
            print("  Loading LoRA adapters (multi-adapter mode — hooks stay live)...")
            from peft import PeftModel as _PeftModel

            # Wrap the already-loaded base_model in-place.
            # PEFT injects LoRA hooks directly into the layer objects we already
            # reference (self.left_layers, etc.) — no re-wiring of layer lists needed.
            if _left_has:
                base_model = _PeftModel.from_pretrained(
                    base_model, _left_dir, adapter_name="left"
                )
                self._has_left_adapter = True
                if _right_has:
                    base_model.load_adapter(_right_dir, adapter_name="right")
                    self._has_right_adapter = True
                    base_model.set_adapter("left")   # default to left
                    print("  Both LoRA adapters loaded (left + right)")
                else:
                    print("  Left LoRA adapter loaded")
            else:
                base_model = _PeftModel.from_pretrained(
                    base_model, _right_dir, adapter_name="right"
                )
                self._has_right_adapter = True
                print("  Right LoRA adapter loaded")

            self.has_lora_adapters = True
            print("  Multi-adapter mode active. LoRA hooks live.")

        # === SHARED LM HEAD (Deduction sigma_D) ===
        self.lm_head = _lm_head_ref

        # === V2 CLAIM 23: CROSS-ATTENTION MODULES ===
        hidden_dim = llama_model.config.hidden_size
        self.n_heads = llama_model.config.num_attention_heads   # for Phi_JS per-head computation
        self.cross_attn_left  = CrossAttentionModule(hidden_dim, config.cross_attn_heads)
        self.cross_attn_right = CrossAttentionModule(hidden_dim, config.cross_attn_heads)

        # === GOVERNOR (manages the wringer) ===
        self.governor = GovernorMLP(hidden_dim=hidden_dim)

        # Move V2 submodules to target device.
        # Cross-attention modules must match base model dtype (bfloat16 on CUDA) because
        # they operate directly on bfloat16 hidden states from the LLM layers.
        # Governor stays float32 — inference.py passes .float() tensors to it, and
        # it's a small MLP where float32 precision is fine.
        _dev = config.device
        _dtype = next(self.lm_head.parameters()).dtype   # bfloat16 on CUDA
        self.cross_attn_left.to(_dev).to(_dtype)
        self.cross_attn_right.to(_dev).to(_dtype)
        self.governor.to(_dev)   # stays float32 — receives .float() inputs from inference

        # Load trained governor weights if available
        import os as _os
        _gov_path = _os.path.join(
            _os.path.dirname(__file__), "trained_models", "governor.pt"
        )
        if _os.path.exists(_gov_path):
            self.governor.load_state_dict(
                torch.load(_gov_path, map_location=_dev, weights_only=True)
            )
            print(f"  Trained governor loaded from {_gov_path}")
        else:
            print(f"  Governor: no trained weights found at {_gov_path} — using random init")

        # Load trained cross-attention weights if available (Phase D — Claim 23)
        _left_ca_path  = _os.path.join(_os.path.dirname(__file__), "trained_models", "cross_attn_left.pt")
        _right_ca_path = _os.path.join(_os.path.dirname(__file__), "trained_models", "cross_attn_right.pt")
        if _os.path.exists(_left_ca_path) and _os.path.exists(_right_ca_path):
            self.cross_attn_left.load_state_dict(
                torch.load(_left_ca_path, map_location=_dev, weights_only=True)
            )
            self.cross_attn_right.load_state_dict(
                torch.load(_right_ca_path, map_location=_dev, weights_only=True)
            )
            print(f"  Trained cross-attention loaded (Claim 23 — Phase D weights)")
        else:
            print(f"  Cross-attention: no trained weights found — using near-zero init (W_O=0)")

        self.base_model = base_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        print("Native THEOS V2 architecture initialized.")
        print(f"  Shared layers:       0-{config.split_layer - 1}")
        print(f"  Left pathway:        {config.split_layer}-{config.split_layer + len(self.left_layers) - 1}")
        print(f"  Right pathway:       {config.split_layer}-{config.split_layer + len(self.right_layers) - 1} (LoRA diverges)")
        print(f"  Cross-attention:     {config.cross_attn_heads} heads per pathway (Claim 23)")
        print(f"  Governor:            4-class MLP, 2d+7 input, learned alpha/beta/gamma (Claims 21,22)")

    # ----------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------

    def _run_layers(
        self,
        hidden_states: torch.Tensor,
        layers: nn.ModuleList,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple] = None,
        cache_position: Optional[torch.Tensor] = None,
        capture_last_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Run hidden states through a sequence of transformer layers.

        If capture_last_attn=True, returns (hidden_states, attn_weights) where
        attn_weights are from the LAST layer — used for Phi_JS computation.
        """
        last_attn = None

        for i, layer in enumerate(layers):
            is_last = (i == len(layers) - 1)

            layer_output = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                cache_position=cache_position,
                output_attentions=(capture_last_attn and is_last),
            )

            if isinstance(layer_output, torch.Tensor):
                hidden_states = layer_output
            elif capture_last_attn and is_last and len(layer_output) >= 2:
                hidden_states = layer_output[0]
                last_attn = layer_output[1]   # [batch, heads, seq, seq]
            else:
                hidden_states = layer_output[0]

        return hidden_states, last_attn

    def _build_position_args(self, hidden_states: torch.Tensor):
        """Build position_ids, cache_position, position_embeddings, causal_mask."""
        seq_len = hidden_states.shape[1]
        device  = hidden_states.device

        position_ids       = torch.arange(seq_len, device=device).unsqueeze(0)
        cache_position     = torch.arange(seq_len, device=device)
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        causal_mask = torch.full(
            (1, 1, seq_len, seq_len),
            fill_value=torch.finfo(hidden_states.dtype).min,
            dtype=hidden_states.dtype,
            device=device,
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)

        return position_ids, cache_position, position_embeddings, causal_mask

    @staticmethod
    def compute_phi_content(
        logits_L: torch.Tensor,    # [batch, seq_len, vocab_size]
        logits_R: torch.Tensor,
    ) -> torch.Tensor:
        """
        Phi_content = JS(P_L || P_R) on last-token output distributions.
        Measures whether pathways are about to say different things.
        Computed before any token is sampled — pure pre-decoded signal.

        Patent: Claim 21.
        """
        p_L = F.softmax(logits_L[:, -1, :].float(), dim=-1)
        p_R = F.softmax(logits_R[:, -1, :].float(), dim=-1)
        m   = 0.5 * (p_L + p_R)

        kl_L = (p_L * ((p_L + 1e-10) / (m + 1e-10)).log()).sum(dim=-1)
        kl_R = (p_R * ((p_R + 1e-10) / (m + 1e-10)).log()).sum(dim=-1)

        js = 0.5 * kl_L + 0.5 * kl_R
        return (js / math.log(2)).clamp(0.0, 1.0).unsqueeze(-1)   # [batch, 1]

    @staticmethod
    def compute_phi_js_hidden(
        h_L: torch.Tensor,   # [batch, seq_len, hidden_dim]
        h_R: torch.Tensor,   # [batch, seq_len, hidden_dim]
        n_heads: int = 32,   # num_attention_heads (Llama 3 8B = 32)
    ) -> torch.Tensor:
        """
        Phi_JS from per-head hidden state divergence (Claim 21).

        Treats hidden_dim as n_heads * head_dim and reshapes to expose per-head
        representations. Applies softmax over head_dim to get probability
        distributions, then computes JS divergence between left and right
        pathway hidden states per head.

        This replaces attention-weight-based Phi_JS when output_attentions=True
        is unavailable (transformers >= 5.x changed decoder layer output format —
        LlamaDecoderLayer now returns a plain tensor, not a tuple).

        Mathematically equivalent: both capture per-head pathway divergence as a
        pre-decoded signal, measured before any token is sampled or generated.

        Patent: Claim 21 — U.S. Application No. 18/919,771
        """
        B, T, D = h_L.shape
        head_dim = D // n_heads

        # Reshape: [batch, seq, n_heads, head_dim]
        p_L = h_L.reshape(B, T, n_heads, head_dim).float()
        p_R = h_R.reshape(B, T, n_heads, head_dim).float()

        # Softmax over head_dim -> valid probability distributions [B, T, n_heads, head_dim]
        p_L = F.softmax(p_L, dim=-1)
        p_R = F.softmax(p_R, dim=-1)

        m = 0.5 * (p_L + p_R)

        def kl(p, q):
            return (p * ((p + 1e-10) / (q + 1e-10)).log()).sum(dim=-1)  # [B, T, n_heads]

        js_per_head = 0.5 * kl(p_L, m) + 0.5 * kl(p_R, m)   # [B, T, n_heads]
        phi_js = js_per_head.mean(dim=(1, 2))                  # [B], avg over seq and heads

        phi_js = (phi_js / math.log(2)).clamp(0.0, 1.0)
        return phi_js.unsqueeze(-1)   # [B, 1]

    # ----------------------------------------------------------------
    # V2 Forward pass — single wringer pass
    # ----------------------------------------------------------------

    def forward_single_pass(
        self,
        input_ids: torch.Tensor,
        past_H_L: Optional[torch.Tensor] = None,   # [batch, prev_seq_len, hidden_dim]
        past_H_R: Optional[torch.Tensor] = None,   # [batch, prev_seq_len, hidden_dim]
    ) -> Dict:
        """
        Execute one wringer pass through the V2 Y-fork architecture.

        V2 Momentary Past (Claim 23):
            When past_H_L / past_H_R are provided, cross-attention is computed
            AFTER the shared layers, BEFORE the pathway layers:
                H_tilde_L = H_shared + CrossAttn_L(H_R^(n-1), H_shared)
                H_tilde_R = H_shared + CrossAttn_R(H_L^(n-1), H_shared)
            Each pathway then processes H_tilde through its own layers.
            This allows each pathway to attend to specific tokens in the other's
            prior state rather than receiving a blurred mean-pooled summary.

        Phi_JS (Claim 21):
            Attention weights are captured from the final shared layer.
            Phi_JS = (1/H) sum_h JS(p^h_L || p^h_R) across shared-layer heads.
            Note: at pass 1 we use the shared layer attentions for both pathways
            (pre-fork). From pass 2 onward, the cross-attention attended states
            give each pathway its own effective attention pattern.

        Returns dict with:
            hidden_L      full left hidden states [batch, seq, hidden_dim]
            hidden_R      full right hidden states
            h_L_pooled    mean-pooled left  [batch, hidden_dim]
            h_R_pooled    mean-pooled right [batch, hidden_dim]
            logits_L      left  token logits
            logits_R      right token logits
            phi_cosine    cosine divergence [batch]
            phi_js        attention divergence [batch, 1]
            phi_content   output distribution divergence [batch, 1]
            phi           composite divergence [batch, 1]  (alpha*cos + beta*js + gamma*content)
        """
        token_embeds = self.embed_tokens(input_ids)

        position_ids, cache_position, position_embeddings, causal_mask = \
            self._build_position_args(token_embeds)

        # === INDUCTION (sigma_I): Shared layers ===
        H_shared, _ = self._run_layers(
            token_embeds, self.shared_layers,
            attention_mask=causal_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            cache_position=cache_position,
        )

        # === ABDUCTION: Fork with V2 cross-attention ===
        if past_H_L is not None and past_H_R is not None:
            # V2 Claim 23: cross-attended fork
            # Left pathway attends to right's PRIOR state
            H_tilde_L = H_shared + self.cross_attn_left(
                H_source=past_H_R.to(H_shared.dtype),
                H_query=H_shared,
            )
            # Right pathway attends to left's PRIOR state
            H_tilde_R = H_shared + self.cross_attn_right(
                H_source=past_H_L.to(H_shared.dtype),
                H_query=H_shared,
            )
        else:
            # Pass 1: no prior state, standard fork
            H_tilde_L = H_shared
            H_tilde_R = H_shared

        # Left pathway — Constructive (sigma_A^L)
        # Activate left LoRA adapter before running pathway layers
        if self.has_lora_adapters and self._has_left_adapter:
            self.base_model.set_adapter("left")
        hidden_L, _ = self._run_layers(
            H_tilde_L.clone(), self.left_layers,
            attention_mask=causal_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            cache_position=cache_position,
        )
        hidden_L = self.left_norm(hidden_L)

        # Right pathway — Adversarial (sigma_A^R)
        # Activate right LoRA adapter before running pathway layers
        if self.has_lora_adapters and self._has_right_adapter:
            self.base_model.set_adapter("right")
        hidden_R, _ = self._run_layers(
            H_tilde_R.clone(), self.right_layers,
            attention_mask=causal_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            cache_position=cache_position,
        )
        hidden_R = self.right_norm(hidden_R)

        # === DEDUCTION (sigma_D): Shared LM head ===
        logits_L = self.lm_head(hidden_L)
        logits_R = self.lm_head(hidden_R)

        # === V2 CLAIM 21: COMPOSITE DIVERGENCE ===
        h_L_pooled  = hidden_L.mean(dim=1)
        h_R_pooled  = hidden_R.mean(dim=1)
        cosine_sim  = F.cosine_similarity(h_L_pooled, h_R_pooled, dim=-1)
        phi_cosine  = (1.0 - cosine_sim)                          # [batch], range [0, 2]

        # Phi_JS: Jensen-Shannon divergence of LEFT vs RIGHT pathway hidden states.
        # (Claim 21 — pre-decoded divergence signal, captured before any token is sampled)
        # Uses per-head hidden state distributions from the final pathway layers.
        # This replaces attention-weight capture which is unavailable in transformers >= 5.x
        # (LlamaDecoderLayer now returns a plain tensor, silently ignoring output_attentions).
        # Mathematically equivalent: both measure per-head pathway divergence pre-decode.
        phi_js = self.compute_phi_js_hidden(hidden_L, hidden_R, self.n_heads)  # [batch, 1]

        phi_content = self.compute_phi_content(logits_L, logits_R)    # [batch, 1]

        # Composite Phi with learned weights (Claim 21)
        phi_composite = self.governor.composite_phi(
            phi_cosine.unsqueeze(-1), phi_js, phi_content
        )

        return {
            "hidden_L":    hidden_L,
            "hidden_R":    hidden_R,
            "h_L_pooled":  h_L_pooled,
            "h_R_pooled":  h_R_pooled,
            "logits_L":    logits_L,
            "logits_R":    logits_R,
            "phi_cosine":  phi_cosine,
            "phi_js":      phi_js,
            "phi_content": phi_content,
            "phi":         phi_composite,        # composite — primary signal
        }

    # ----------------------------------------------------------------
    # Generation
    # ----------------------------------------------------------------

    def generate_through_pathway(
        self,
        input_ids: torch.Tensor,
        pathway: str = "left",
        past_H_L: Optional[torch.Tensor] = None,   # full hidden state [batch, seq, hidden_dim]
        past_H_R: Optional[torch.Tensor] = None,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
    ) -> Tuple[str, torch.Tensor, torch.Tensor]:
        """
        Generate text through one pathway via the Y-fork (no KV cache).

        V2: returns (generated_text, h_pooled, H_full) where H_full is the
        full hidden state sequence [batch, seq, hidden_dim] — needed for
        cross-attention in the next wringer pass (Claim 23).

        The Momentary Past is injected every generation step so all generated
        tokens are conditioned on the previous pass's full hidden state.
        """
        device = input_ids.device
        generated_ids = input_ids.clone()
        last_result = None

        with torch.inference_mode():
            for _ in range(max_new_tokens):
                result = self.forward_single_pass(
                    generated_ids,
                    past_H_L=past_H_L,
                    past_H_R=past_H_R,
                )
                last_result = result

                logits = result["logits_L"] if pathway == "left" else result["logits_R"]
                next_logits = logits[:, -1, :].float()

                if temperature > 0:
                    next_logits = next_logits / temperature
                    probs = F.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = next_logits.argmax(dim=-1, keepdim=True)

                generated_ids = torch.cat([generated_ids, next_token], dim=1)

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        new_ids = generated_ids[:, input_ids.shape[1]:]
        text = self.tokenizer.decode(new_ids[0], skip_special_tokens=True)

        # Return pooled vector AND full hidden state sequence
        H_full  = last_result["hidden_L"] if pathway == "left" else last_result["hidden_R"]
        h_pooled = last_result["h_L_pooled"] if pathway == "left" else last_result["h_R_pooled"]

        return text, h_pooled, H_full

    def generate_from_pathway(
        self,
        input_ids: torch.Tensor,
        pathway: str = "left",
        max_new_tokens: int = 200,
        temperature: float = 0.7,
    ) -> Tuple[str, torch.Tensor]:
        """
        Fast KV-cached generation via base_model.generate().
        Returns (text, hidden_states) — used for first-pass generation
        where no Momentary Past is needed.
        """
        # Set the correct LoRA adapter before generation
        if self.has_lora_adapters:
            if pathway == "left" and self._has_left_adapter:
                self.base_model.set_adapter("left")
            elif pathway == "right" and self._has_right_adapter:
                self.base_model.set_adapter("right")

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=(temperature > 0),
        )
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.inference_mode():
            out_ids = self.base_model.generate(input_ids, **gen_kwargs)

        new_ids = out_ids[:, input_ids.shape[1]:]
        text = self.tokenizer.decode(new_ids[0], skip_special_tokens=True)

        with torch.inference_mode():
            result = self.forward_single_pass(out_ids[:, :512])

        hidden = result["hidden_L"] if pathway == "left" else result["hidden_R"]
        return text, hidden
