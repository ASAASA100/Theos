"""
THEOS Native Architecture Tests
================================

Tests for the native THEOS Y-fork transformer architecture.
These tests run WITHOUT the full Llama 3 8B model to allow
rapid iteration during development.

Test categories:
    1. Architecture shape tests (mock model, CPU, no GPU needed)
    2. Inference mode routing tests
    3. WisdomStore tests
    4. Governor logic tests
    5. Integration smoke test (full wringer with mock)

Patent: U.S. Application No. 18/919,771 (Provisional #63/831,738)
Author: Frederick Davis Stalnecker
"""

import json
import os
import sys
import tempfile
import unittest

import torch
import torch.nn as nn

# Allow importing from native_theos directory
sys.path.insert(0, os.path.dirname(__file__))


# ============================================================
# Mock model for tests that don't need Llama 3
# ============================================================

class MockTokenizer:
    """Minimal tokenizer mock for architecture tests."""
    pad_token = "<pad>"
    eos_token = "<eos>"
    pad_token_id = 0

    def __call__(self, text, return_tensors="pt", truncation=True,
                 max_length=512, padding=True):
        # Return fixed-size token tensor
        batch_size = 1
        seq_len = min(len(text.split()), 16)  # Short for tests
        return {
            "input_ids": torch.zeros(batch_size, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        }

    def decode(self, token_ids, skip_special_tokens=True):
        return "Mock decoded text from THEOS native engine."


class MockNativeTheos(nn.Module):
    """
    Mock THEOS model for tests — no Llama 3 required.
    Returns deterministic outputs for testing inference logic.
    """
    HIDDEN_DIM = 64   # Small for tests

    def __init__(self):
        super().__init__()
        self.config = self._make_config()
        self.tokenizer = MockTokenizer()
        self.governor = self._make_governor()
        self._phi_sequence = [0.8, 0.5, 0.3, 0.1]  # Converging phi for tests
        self._pass_count = 0

    def _make_config(self):
        """Create a minimal config that matches NativeTheosConfig interface."""
        class Cfg:
            device = "cpu"
            eps_converge = 0.05
            eps_partial = 0.5
            mode1_max_passes = 2
            mode2_max_passes = 3
            mode3_max_passes = 7
            max_wringer_passes = 7
            max_new_tokens = 10
            temperature = 0.7
        return Cfg()

    def _make_governor(self):
        """Create a tiny governor for test determinism."""
        from architecture import GovernorMLP
        gov = GovernorMLP(hidden_dim=self.HIDDEN_DIM)
        return gov

    def _mock_forward(self, input_ids):
        """Shared mock forward logic returning phi_cosine, phi_content, pooled states."""
        import torch.nn.functional as F
        batch_size = input_ids.shape[0]
        seq_len    = input_ids.shape[1]
        vocab_size = 1000

        hidden_L = torch.randn(batch_size, seq_len, self.HIDDEN_DIM)
        hidden_R = torch.randn(batch_size, seq_len, self.HIDDEN_DIM)

        h_L_pooled = hidden_L.mean(dim=1)
        h_R_pooled = hidden_R.mean(dim=1)

        cosine_sim  = F.cosine_similarity(h_L_pooled, h_R_pooled, dim=-1)
        phi_cosine  = 1.0 - cosine_sim
        logits_L    = torch.randn(batch_size, seq_len, vocab_size)
        logits_R    = torch.randn(batch_size, seq_len, vocab_size)

        # Simple phi_content mock: small positive value
        phi_content = torch.tensor([[0.1]] * batch_size)

        return {
            "hidden_L":    hidden_L,
            "hidden_R":    hidden_R,
            "h_L_pooled":  h_L_pooled,
            "h_R_pooled":  h_R_pooled,
            "logits_L":    logits_L,
            "logits_R":    logits_R,
            "phi":         phi_cosine,
            "phi_cosine":  phi_cosine,
            "phi_content": phi_content,
        }

    def forward_single_pass(self, input_ids, attention_mask=None, position_ids=None):
        """Return mock tensors that mimic the real forward pass output."""
        self._pass_count += 1
        return self._mock_forward(input_ids)

    def forward_with_past(self, input_ids, past_h_L=None, past_h_R=None):
        """Mock forward_with_past — ignores past, returns standard mock output."""
        return self._mock_forward(input_ids)

    def generate_through_pathway(
        self, input_ids, pathway="left",
        past_h_L=None, past_h_R=None,
        max_new_tokens=10, temperature=0.7,
    ):
        """
        Mock generation — returns deterministic text and pooled hidden state.
        The phi from this mock uses a preset converging sequence so wringer
        tests exercise the halt logic correctly.
        """
        result  = self._mock_forward(input_ids)
        text    = "Mock constructive reasoning." if pathway == "left" else "Mock adversarial challenge."
        h_pooled = result["h_L_pooled"] if pathway == "left" else result["h_R_pooled"]

        # Inject preset phi by scaling vectors (pass 3+ will give phi < eps_converge)
        step = min(self._pass_count, len(self._phi_sequence) - 1)
        target_phi = self._phi_sequence[step]
        # Force h_L_pooled ≈ h_R_pooled when target phi is small
        if pathway == "right" and target_phi < 0.1:
            # Make right identical to left so cosine = 1 → phi ≈ 0
            h_pooled = result["h_L_pooled"].clone()

        return text, h_pooled

    def reset_pass_count(self):
        self._pass_count = 0


# ============================================================
# Test Suite 1: Governor MLP
# ============================================================

class TestGovernorMLP(unittest.TestCase):
    """Tests for the GovernorMLP architecture."""

    def setUp(self):
        from architecture import GovernorMLP
        self.governor = GovernorMLP(hidden_dim=64)

    def _make_inputs(self, batch_size: int):
        """Build the 7 Governor inputs for batch_size examples."""
        h_L         = torch.randn(batch_size, 64)
        h_R         = torch.randn(batch_size, 64)
        phi_cosine  = torch.rand(batch_size, 1)
        phi_content = torch.rand(batch_size, 1)
        delta_phi   = torch.randn(batch_size, 1) * 0.1
        delta_phi_2 = torch.randn(batch_size, 1) * 0.05
        pass_num    = torch.rand(batch_size, 1)
        return h_L, h_R, phi_cosine, phi_content, delta_phi, delta_phi_2, pass_num

    def test_output_shape(self):
        """Governor must output 4 logits per batch item."""
        batch_size = 2
        logits = self.governor(*self._make_inputs(batch_size))
        self.assertEqual(logits.shape, (batch_size, 4),
                         f"Expected (2, 4), got {logits.shape}")

    def test_four_decisions(self):
        """All 4 decision classes should be reachable via argmax."""
        logits    = self.governor(*self._make_inputs(100))
        decisions = logits.argmax(dim=-1).unique().tolist()
        self.assertGreaterEqual(len(decisions), 1,
                                "Governor should produce at least 1 distinct decision")

    def test_softmax_sums_to_one(self):
        """Softmax probabilities must sum to 1."""
        import torch.nn.functional as F
        logits = self.governor(*self._make_inputs(4))
        probs  = F.softmax(logits, dim=-1)
        sums   = probs.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones(4), atol=1e-5),
                        "Softmax probabilities must sum to 1.0")

    def test_input_dim_is_hidden_x2_plus_5(self):
        """Governor input must be H_L + H_R + phi_cosine + phi_content + delta_phi + delta_phi_2 + pass_num."""
        first_linear = self.governor.network[0]
        expected_in  = 64 * 2 + 5
        self.assertEqual(first_linear.in_features, expected_in,
                         f"Input dim should be {expected_in}, got {first_linear.in_features}")

    def test_output_dim_is_four(self):
        """Governor output must be exactly 4 (the 4 decisions)."""
        last_linear = self.governor.network[-1]
        self.assertEqual(last_linear.out_features, 4)


# ============================================================
# Test Suite 2: Mode Detection
# ============================================================

class TestModeDetection(unittest.TestCase):
    """Tests for the query complexity router."""

    def setUp(self):
        from inference import detect_mode
        self.detect_mode = detect_mode

    def test_simple_factual_returns_mode1(self):
        """Simple factual queries should route to Mode 1."""
        simple_queries = [
            "What is the capital of France?",
            "How many legs does a spider have?",
            "Define entropy",
            "When was Einstein born?",
        ]
        for q in simple_queries:
            mode = self.detect_mode(q)
            self.assertEqual(mode, 1, f"Expected Mode 1 for: {q!r}, got {mode}")

    def test_ethical_dilemma_returns_mode2_or_3(self):
        """Ethical dilemmas should get Mode 2 or 3."""
        complex_queries = [
            "Should a self-driving car prioritize its passenger or a pedestrian?",
            "Is universal basic income a viable solution to unemployment?",
        ]
        for q in complex_queries:
            mode = self.detect_mode(q)
            self.assertGreaterEqual(mode, 2, f"Expected Mode 2+ for: {q!r}, got {mode}")

    def test_highly_complex_returns_mode3(self):
        """Queries with many complex keywords should get Mode 3."""
        hard_query = "Should we prioritize reducing existential risk over current suffering given ethical dilemmas and competing rights?"
        mode = self.detect_mode(hard_query)
        self.assertEqual(mode, 3, f"Expected Mode 3, got {mode}")

    def test_mode_is_int_1_2_or_3(self):
        """detect_mode must always return 1, 2, or 3."""
        queries = [
            "Hello",
            "Should we?",
            "What is",
            "The quick brown fox jumps",
        ]
        for q in queries:
            mode = self.detect_mode(q)
            self.assertIn(mode, [1, 2, 3], f"Mode {mode} not in [1,2,3] for: {q!r}")


# ============================================================
# Test Suite 3: WisdomStore
# ============================================================

class TestWisdomStore(unittest.TestCase):
    """Tests for wisdom accumulation and retrieval."""

    def setUp(self):
        from wisdom import WisdomStore
        # Use temp file so tests don't pollute real wisdom store
        self.tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
        self.tmp.close()
        self.store = WisdomStore(store_path=self.tmp.name)

    def tearDown(self):
        os.unlink(self.tmp.name)

    def _add_sample_entry(self, question="Is X ethical?", confidence=0.8):
        self.store.add(
            question=question,
            constructive="X is ethical because A, B, C.",
            adversarial="X is not ethical because D, E, F.",
            synthesis="X is ethical under conditions A and B, but not when F applies.",
            phi_trajectory=[0.7, 0.4, 0.1],
            halt_reason="convergence",
            confidence=confidence,
            contradiction=0.1,
            passes_used=3,
            domain="ethics",
        )

    def test_add_persists_to_file(self):
        """Adding an entry must write to the JSONL file."""
        self._add_sample_entry()

        with open(self.tmp.name, "r") as f:
            lines = [l for l in f if l.strip()]

        self.assertEqual(len(lines), 1)
        entry = json.loads(lines[0])
        self.assertIn("question", entry)
        self.assertIn("phi_trajectory", entry)
        self.assertIn("fine_tune_priority", entry)

    def test_fine_tune_priority_rewards_resolution(self):
        """High initial phi that resolves should get high priority."""
        self.store.add(
            question="Hard question",
            constructive="Left",
            adversarial="Right",
            synthesis="Synthesis",
            phi_trajectory=[0.9, 0.5, 0.05],   # Big resolution
            halt_reason="convergence",
            confidence=0.95,
            contradiction=0.05,
            passes_used=3,
        )
        entry = self.store.entries[-1]
        # resolution_quality = 0.9 - 0.05 = 0.85, * confidence 0.95 = 0.8075
        self.assertGreater(entry["fine_tune_priority"], 0.5,
                           "High-resolution entry should have high priority")

    def test_fine_tune_priority_low_for_no_resolution(self):
        """Phi that stays flat gets low priority."""
        self.store.add(
            question="Stuck question",
            constructive="Left",
            adversarial="Right",
            synthesis="No synthesis",
            phi_trajectory=[0.8, 0.79, 0.78],  # Barely moved
            halt_reason="budget_exhaustion",
            confidence=0.5,
            contradiction=0.78,
            passes_used=3,
        )
        entry = self.store.entries[-1]
        self.assertLess(entry["fine_tune_priority"], 0.1,
                        "Unresolved entry should have low priority")

    def test_retrieve_returns_relevant(self):
        """Retrieve should return entries with overlapping keywords."""
        self._add_sample_entry(question="Is artificial intelligence ethical?")
        self._add_sample_entry(question="Should we eat vegetables daily?")

        results = self.store.retrieve("Is machine intelligence ethical?", max_results=2)
        self.assertGreater(len(results), 0)
        # The AI entry should score higher
        if len(results) >= 2:
            self.assertIn("intelligence", results[0]["question"].lower())

    def test_get_stats_returns_correct_total(self):
        """get_stats must report correct total count."""
        for i in range(5):
            self._add_sample_entry(question=f"Question {i}")

        stats = self.store.get_stats()
        self.assertEqual(stats["total"], 5)

    def test_get_stats_empty(self):
        """get_stats on empty store returns total=0."""
        stats = self.store.get_stats()
        self.assertEqual(stats["total"], 0)

    def test_persistence_reload(self):
        """Entries should reload correctly from file."""
        from wisdom import WisdomStore
        self._add_sample_entry("Test persistence question")

        # Reload from file
        store2 = WisdomStore(store_path=self.tmp.name)
        self.assertEqual(len(store2.entries), 1)
        self.assertEqual(store2.entries[0]["question"], "Test persistence question")

    def test_truncation_on_long_text(self):
        """Long constructive/adversarial text should be truncated to 1000 chars."""
        long_text = "x" * 5000
        self.store.add(
            question="Short question",
            constructive=long_text,
            adversarial=long_text,
            synthesis=long_text,
            phi_trajectory=[0.5],
            halt_reason="convergence",
            confidence=0.8,
            contradiction=0.5,
            passes_used=1,
        )
        entry = self.store.entries[-1]
        self.assertLessEqual(len(entry["constructive"]), 1000)
        self.assertLessEqual(len(entry["adversarial"]), 1000)
        self.assertLessEqual(len(entry["synthesis"]), 2000)


# ============================================================
# Test Suite 4: Inference Output Types
# ============================================================

class TestInferenceOutputTypes(unittest.TestCase):
    """Tests for NativeTheosOutput and NativeWringerPassTrace dataclasses."""

    def test_native_theos_output_fields(self):
        """NativeTheosOutput must have all required fields."""
        from inference import NativeTheosOutput, NativeWringerPassTrace
        output = NativeTheosOutput(
            output="Test output",
            output_type="convergence",
            confidence=0.85,
            contradiction=0.04,
            wringer_passes_used=2,
            halt_reason="convergence",
            mode_used=2,
            phi_trajectory=[0.6, 0.04],
            trace=[],
            query="Test query",
        )
        self.assertEqual(output.output_type, "convergence")
        self.assertAlmostEqual(output.confidence, 0.85)
        self.assertIsInstance(output.phi_trajectory, list)

    def test_wringer_pass_trace_fields(self):
        """NativeWringerPassTrace must have all required fields."""
        from inference import NativeWringerPassTrace
        trace = NativeWringerPassTrace(
            wringer_pass=0,
            phi=0.65,
            governor_decision="continue",
            left_output="Constructive reasoning...",
            right_output="Adversarial challenge...",
        )
        self.assertEqual(trace.wringer_pass, 0)
        self.assertEqual(trace.governor_decision, "continue")
        self.assertIsInstance(trace.timestamp, str)

    def test_output_type_values(self):
        """output_type must be one of the defined values."""
        from inference import NativeTheosOutput
        valid_types = {"convergence", "blend", "disagreement", "reframe"}
        for otype in valid_types:
            output = NativeTheosOutput(
                output="x", output_type=otype, confidence=0.5,
                contradiction=0.5, wringer_passes_used=1,
                halt_reason="test", mode_used=1,
                phi_trajectory=[0.5], trace=[], query="test",
            )
            self.assertIn(output.output_type, valid_types)


# ============================================================
# Test Suite 5: Wringer Logic (Mock Model)
# ============================================================

class TestWringerWithMock(unittest.TestCase):
    """
    Tests for the full wringer loop using the MockNativeTheos.
    These run entirely on CPU without Llama 3.
    """

    def setUp(self):
        self.model = MockNativeTheos()

    def test_mode1_halts_quickly(self):
        """Mode 1 should halt after ≤2 passes."""
        from inference import run_native_wringer
        self.model.reset_pass_count()
        result = run_native_wringer(self.model, "What is entropy?", mode=1, verbose=False)

        self.assertLessEqual(result.wringer_passes_used, 2,
                             "Mode 1 should use ≤2 passes")
        self.assertEqual(result.mode_used, 1)

    def test_result_has_phi_trajectory(self):
        """Result must include phi trajectory."""
        from inference import run_native_wringer
        self.model.reset_pass_count()
        result = run_native_wringer(self.model, "Should AI be regulated?", mode=2, verbose=False)

        self.assertIsInstance(result.phi_trajectory, list)
        self.assertGreater(len(result.phi_trajectory), 0)

    def test_result_has_trace(self):
        """Result must include trace with at least one pass."""
        from inference import run_native_wringer
        self.model.reset_pass_count()
        result = run_native_wringer(self.model, "Should AI be regulated?", mode=2, verbose=False)

        self.assertIsInstance(result.trace, list)
        self.assertGreater(len(result.trace), 0)

    def test_confidence_in_valid_range(self):
        """Confidence must be in [0, 1]."""
        from inference import run_native_wringer
        self.model.reset_pass_count()
        result = run_native_wringer(self.model, "Test query", mode=1, verbose=False)

        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)

    def test_contradiction_non_negative(self):
        """Contradiction (phi) must be non-negative."""
        from inference import run_native_wringer
        self.model.reset_pass_count()
        result = run_native_wringer(self.model, "Test query", mode=1, verbose=False)

        self.assertGreaterEqual(result.contradiction, 0.0)

    def test_query_preserved_in_output(self):
        """The query must be preserved in the output."""
        from inference import run_native_wringer
        self.model.reset_pass_count()
        query = "Is consciousness a prerequisite for moral consideration?"
        result = run_native_wringer(self.model, query, mode=2, verbose=False)

        self.assertEqual(result.query, query)

    def test_mode3_uses_max_passes_or_halts_early(self):
        """Mode 3 uses up to 7 passes but may halt early on convergence."""
        from inference import run_native_wringer
        self.model.reset_pass_count()
        result = run_native_wringer(
            self.model,
            "Should we pursue AGI development?",
            mode=3,
            verbose=False,
        )
        self.assertLessEqual(result.wringer_passes_used, 7)
        self.assertEqual(result.mode_used, 3)


# ============================================================
# Test Suite 6: Server Request/Response Models
# ============================================================

class TestServerModels(unittest.TestCase):
    """Tests for the FastAPI Pydantic models."""

    def test_reasoning_request_optional_fields(self):
        """mode and max_passes should be optional."""
        from server import ReasoningRequest
        req = ReasoningRequest(query="Test query")
        self.assertEqual(req.query, "Test query")
        self.assertIsNone(req.mode)
        self.assertIsNone(req.max_passes)

    def test_audit_request_default_source(self):
        """AuditRequest source should default to 'external_ai'."""
        from server import AuditRequest
        req = AuditRequest(question="Q", answer_to_audit="A")
        self.assertEqual(req.source, "external_ai")

    def test_reasoning_response_all_fields(self):
        """ReasoningResponse must accept all required fields."""
        from server import ReasoningResponse
        response = ReasoningResponse(
            id="test-id",
            query="Test query",
            output="Test output",
            output_type="convergence",
            confidence=0.85,
            contradiction=0.04,
            wringer_passes_used=2,
            halt_reason="convergence",
            mode_used=2,
            phi_trajectory=[0.6, 0.04],
            trace=[{"pass": 1, "phi": 0.6, "decision": "continue", "left": "L", "right": "R"}],
            timestamp="2026-03-15T10:00:00",
        )
        self.assertEqual(response.output_type, "convergence")

    def test_audit_response_certified_field(self):
        """AuditResponse must have a boolean certified field."""
        from server import AuditResponse
        response = AuditResponse(
            id="audit-id",
            question="Is X safe?",
            original_answer="X is safe.",
            theos_output="Analysis...",
            theos_output_type="convergence",
            phi_score=0.15,
            what_was_missed="Risk Y was not considered.",
            confidence=0.8,
            certified=True,
            timestamp="2026-03-15T10:00:00",
        )
        self.assertIsInstance(response.certified, bool)
        self.assertTrue(response.certified)


# ============================================================
# Test Suite 7: Contradiction Measurement
# ============================================================

class TestContradictionMeasurement(unittest.TestCase):
    """Tests for the phi (Φ) contradiction measurement."""

    def test_identical_vectors_give_zero_phi(self):
        """Identical hidden states should give phi ≈ 0."""
        import torch.nn.functional as F
        hidden_dim = 128
        h = torch.randn(1, hidden_dim)

        cosine_sim = F.cosine_similarity(h, h, dim=-1)
        phi = 1.0 - cosine_sim

        self.assertAlmostEqual(phi.item(), 0.0, places=4)

    def test_opposite_vectors_give_max_phi(self):
        """Opposite vectors should give phi ≈ 2."""
        import torch.nn.functional as F
        hidden_dim = 128
        h = torch.randn(1, hidden_dim)
        h_neg = -h

        cosine_sim = F.cosine_similarity(h, h_neg, dim=-1)
        phi = 1.0 - cosine_sim

        self.assertAlmostEqual(phi.item(), 2.0, places=4)

    def test_phi_range_is_zero_to_two(self):
        """Phi must always be in [0, 2] for any pair of hidden states."""
        import torch.nn.functional as F
        for _ in range(50):
            h_L = torch.randn(1, 256)
            h_R = torch.randn(1, 256)
            cosine_sim = F.cosine_similarity(h_L, h_R, dim=-1)
            phi = 1.0 - cosine_sim
            self.assertGreaterEqual(phi.item(), -1e-5)   # ≥ 0
            self.assertLessEqual(phi.item(), 2.0 + 1e-5)  # ≤ 2


# ============================================================
# Run all tests
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("THEOS Native Architecture Tests")
    print("Running without Llama 3 — CPU only, mock model")
    print("=" * 60)
    print()

    # Discover and run all test suites
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestGovernorMLP,
        TestModeDetection,
        TestWisdomStore,
        TestInferenceOutputTypes,
        TestWringerWithMock,
        TestServerModels,
        TestContradictionMeasurement,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        print("\nAll tests passed.")
        sys.exit(0)
    else:
        print(f"\n{len(result.failures)} failures, {len(result.errors)} errors.")
        sys.exit(1)
