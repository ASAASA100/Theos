"""
Phi_JS Diagnostic — run on RunPod to determine why phi_JS = 0.

Usage:
    cd /workspace/THEOS/native_theos
    python phi_js_diag.py

Checks:
1. Does output_attentions=True actually return attn weights with eager mode?
2. Are attn_L and attn_R different after LoRA fine-tuning?
3. Is JS divergence being computed correctly?
"""
import sys, os, torch
sys.path.insert(0, os.path.dirname(__file__))
from architecture import NativeTheos, NativeTheosConfig, compute_phi_js
from peft import PeftModel

print("Loading model (this takes ~2 min)...")
config = NativeTheosConfig(device="cuda")
model = NativeTheos(config)

# Load trained adapters
out_dir  = os.path.join(os.path.dirname(__file__), "trained_models")
left_dir  = os.path.join(out_dir, "left_pathway")
right_dir = os.path.join(out_dir, "right_pathway")

if os.path.exists(os.path.join(left_dir, "adapter_config.json")):
    model.base_model = PeftModel.from_pretrained(model.base_model, left_dir, adapter_name="left")
    model._has_left_adapter = True
    if os.path.exists(os.path.join(right_dir, "adapter_config.json")):
        model.base_model.load_adapter(right_dir, adapter_name="right")
        model._has_right_adapter = True
        model.base_model.set_adapter("left")
    model.has_lora_adapters = True
    print("LoRA adapters loaded.")
else:
    print("WARNING: No trained adapters found. Testing with base weights only.")

q = "Should terminally ill patients have unrestricted access to experimental treatments?"
inputs = model.tokenizer(q, return_tensors="pt", truncation=True, max_length=64).to("cuda")

model.eval()

# -------------------------------------------------------
# TEST 1: Full forward pass — check all three Phi values
# -------------------------------------------------------
print("\n=== TEST 1: Full forward pass ===")
with torch.no_grad():
    result = model.forward_single_pass(inputs["input_ids"])

print(f"phi_cosine:  {result['phi_cosine'].mean().item():.6f}  (should be > 0)")
print(f"phi_js:      {result['phi_js'].mean().item():.6f}  (may be 0 — diagnosing below)")
print(f"phi_content: {result['phi_content'].mean().item():.6f}  (should be > 0)")
print(f"phi_comp:    {result['phi'].mean().item():.6f}")

# -------------------------------------------------------
# TEST 2: Direct attention capture from pathway layers
# -------------------------------------------------------
print("\n=== TEST 2: Direct attention capture ===")
with torch.no_grad():
    h = model.embed_tokens(inputs["input_ids"])
    pos_ids, cache_pos, pos_emb, causal_mask = model._build_position_args(h)

    # Shared layers
    H_shared, shared_attn = model._run_layers(
        h, model.shared_layers,
        attention_mask=causal_mask,
        position_ids=pos_ids,
        position_embeddings=pos_emb,
        cache_position=cache_pos,
        capture_last_attn=True,
    )
    print(f"shared_attn is None: {shared_attn is None}")
    if shared_attn is not None:
        print(f"shared_attn.shape:   {shared_attn.shape}")

    # Left pathway
    if model.has_lora_adapters and model._has_left_adapter:
        model.base_model.set_adapter("left")
    _, attn_L = model._run_layers(
        H_shared.clone(), model.left_layers,
        attention_mask=causal_mask,
        position_ids=pos_ids,
        position_embeddings=pos_emb,
        cache_position=cache_pos,
        capture_last_attn=True,
    )
    print(f"attn_L is None: {attn_L is None}")
    if attn_L is not None:
        print(f"attn_L.shape:   {attn_L.shape}")
        print(f"attn_L.sum():   {attn_L.sum().item():.4f}")

    # Right pathway
    if model.has_lora_adapters and model._has_right_adapter:
        model.base_model.set_adapter("right")
    _, attn_R = model._run_layers(
        H_shared.clone(), model.right_layers,
        attention_mask=causal_mask,
        position_ids=pos_ids,
        position_embeddings=pos_emb,
        cache_position=cache_pos,
        capture_last_attn=True,
    )
    print(f"attn_R is None: {attn_R is None}")
    if attn_R is not None:
        print(f"attn_R.shape:   {attn_R.shape}")
        print(f"attn_R.sum():   {attn_R.sum().item():.4f}")

# -------------------------------------------------------
# TEST 3: If we have both attention tensors, compute JS
# -------------------------------------------------------
if attn_L is not None and attn_R is not None:
    print("\n=== TEST 3: JS divergence computation ===")
    phi_js_direct = compute_phi_js(attn_L, attn_R)
    print(f"phi_js_direct:  {phi_js_direct.mean().item():.8f}")
    print(f"attn_L vs attn_R max diff: {(attn_L - attn_R).abs().max().item():.8f}")
    print(f"attn_L vs attn_R mean diff: {(attn_L - attn_R).abs().mean().item():.8f}")
    if phi_js_direct.mean().item() == 0.0:
        print("\nDIAGNOSIS: attn_L and attn_R are numerically identical.")
        print("This means LoRA fine-tuning has not diverged attention patterns sufficiently.")
        print("Fix: more training data, more epochs, or check that set_adapter() is working.")
    else:
        print("\nDIAGNOSIS: phi_JS is non-zero. Attention patterns are diverging correctly.")
else:
    print("\n=== TEST 3: SKIPPED — one or both attention tensors is None ===")
    print("DIAGNOSIS: output_attentions=True is not returning attention weights.")
    print("Check transformers version and attn_implementation setting.")
    import transformers
    print(f"transformers version: {transformers.__version__}")

    # Check what the layer actually returns
    print("\n=== TEST 3b: Raw layer output inspection ===")
    with torch.no_grad():
        H_shared2, _ = model._run_layers(
            h, model.shared_layers,
            attention_mask=causal_mask,
            position_ids=pos_ids,
            position_embeddings=pos_emb,
            cache_position=cache_pos,
            capture_last_attn=False,
        )
        # Manually call last layer with output_attentions=True
        last_layer = model.left_layers[-1]
        raw_output = last_layer(
            H_shared2,
            attention_mask=causal_mask,
            position_ids=pos_ids,
            position_embeddings=pos_emb,
            cache_position=cache_pos,
            output_attentions=True,
        )
        print(f"raw_output type: {type(raw_output)}")
        print(f"raw_output len: {len(raw_output)}")
        for i, o in enumerate(raw_output):
            if o is None:
                print(f"  raw_output[{i}]: None")
            elif isinstance(o, torch.Tensor):
                print(f"  raw_output[{i}]: Tensor {o.shape}")
            else:
                print(f"  raw_output[{i}]: {type(o)}")
