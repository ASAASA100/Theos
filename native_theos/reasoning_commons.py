"""
THEOS Reasoning Commons
========================

The shared dialectical wisdom layer — a curated repository of reasoning that
has been through the THEOS wringer.

Each entry in the Commons represents a distilled insight:
    - A question that generated real dialectical tension (Phi > eps_converge)
    - The constructive and adversarial positions (the thesis and antithesis)
    - The synthesis (what the wringer resolved)
    - The Phi trajectory (the intellectual history of the resolution)
    - The certification score for any external AI that addressed the same question

The Commons is cumulative and quality-gated:
    - Only entries with confidence > 0.6 are admitted
    - Entries are indexed by Phi trajectory shape (not just keywords)
    - Any THEOS instance or certified AI can contribute
    - Retrieval uses TF-IDF cosine similarity for domain-aware lookup

This is the Reasoning equivalent of a peer-reviewed journal:
    - Peer review = the THEOS wringer (both engines must have contested it)
    - Open access = any AI can query
    - Cumulative = each new entry is informed by existing entries

Patent: U.S. Application No. 18/919,771 (Provisional #63/831,738)
Author: Frederick Davis Stalnecker
"""

import json
import math
import os
import re
import uuid
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional


# ============================================================
# TF-IDF SIMILARITY (no external dependencies)
# ============================================================

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z]+", text.lower())


def _tfidf_cosine(query: str, document: str, corpus_docs: List[str]) -> float:
    """Cosine similarity using TF-IDF weighting."""
    all_docs = corpus_docs + [query, document]
    N = len(all_docs)

    # Document frequency for each term
    df: Counter = Counter()
    tokenized = [_tokenize(d) for d in all_docs]
    for tokens in tokenized:
        for term in set(tokens):
            df[term] += 1

    def tfidf_vec(tokens: List[str]) -> Dict[str, float]:
        tf = Counter(tokens)
        total = len(tokens) or 1
        vec = {}
        for term, count in tf.items():
            t = count / total
            idf = math.log((N + 1) / (df[term] + 1)) + 1
            vec[term] = t * idf
        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {k: v / norm for k, v in vec.items()}

    q_tokens = _tokenize(query)
    d_tokens = _tokenize(document)
    if not q_tokens or not d_tokens:
        return 0.0

    q_vec = tfidf_vec(q_tokens)
    d_vec = tfidf_vec(d_tokens)

    common = set(q_vec) & set(d_vec)
    return sum(q_vec[t] * d_vec[t] for t in common)


# ============================================================
# PHI TRAJECTORY SIMILARITY (for quality-aware retrieval)
# ============================================================

def _phi_trajectory_similarity(traj_a: List[float], traj_b: List[float]) -> float:
    """
    Simplified DTW-like trajectory similarity.

    Two trajectories that converge from high initial Phi are more similar
    than two that start low and stay low. This prioritizes high-quality
    (high initial tension, resolved) entries over low-tension entries.
    """
    if not traj_a or not traj_b:
        return 0.0

    # Normalize lengths by padding shorter trajectory
    max_len = max(len(traj_a), len(traj_b))
    a = traj_a + [traj_a[-1]] * (max_len - len(traj_a))
    b = traj_b + [traj_b[-1]] * (max_len - len(traj_b))

    # Mean absolute difference (lower = more similar)
    mad = sum(abs(x - y) for x, y in zip(a, b)) / max_len
    return max(0.0, 1.0 - mad * 5.0)   # Scale: 0.2 diff -> 0 similarity


# ============================================================
# CERTIFICATION SCORING
# ============================================================

def compute_certification_score(
    theos_phi: float,
    theos_confidence: float,
    theos_halt_reason: str,
    external_answer: str,
    theos_synthesis: str,
    theos_adversarial: str,
) -> Dict[str, Any]:
    """
    Compute a THEOS Certification Score for an external AI's answer.

    Scoring rubric (100 points total):
        40 pts — Dialectical completeness:
                 Did the external answer acknowledge the opposing framing?
                 Measured by: word overlap with THEOS adversarial output.

        30 pts — Resolution quality:
                 Did the external answer reach a comparable synthesis?
                 Measured by: similarity between external answer and THEOS synthesis.

        30 pts — Epistemic honesty:
                 Did the external answer acknowledge uncertainty where THEOS found
                 irreducible tension (halt_reason = irreducible_uncertainty)?
                 Full points if no irreducible tension; partial if acknowledged.

    Returns:
        certification_score: 0-100 integer
        grade: A/B/C/D/F
        gaps: list of specific gaps found
        verdict: one-sentence summary
    """
    gaps = []
    score = 0

    # --- Dialectical completeness (40 pts) ---
    adv_words = set(_tokenize(theos_adversarial))
    ext_words = set(_tokenize(external_answer))
    # Stop words to ignore
    stop = {"the", "a", "an", "is", "it", "in", "of", "to", "and", "or", "but",
            "that", "this", "with", "for", "on", "at", "by", "from", "be", "are",
            "was", "were", "have", "has", "had", "not", "no", "so", "if", "as"}
    adv_content = adv_words - stop
    overlap = len(adv_content & ext_words) / max(len(adv_content), 1)
    dialectical_score = min(40, int(overlap * 80))  # generous scaling
    score += dialectical_score
    if dialectical_score < 20:
        gaps.append(
            "Missing adversarial framing — THEOS found significant opposing "
            "considerations that were not addressed."
        )

    # --- Resolution quality (30 pts) ---
    # Use simple token overlap between external answer and THEOS synthesis
    syn_words = set(_tokenize(theos_synthesis)) - stop
    resolution_overlap = len(syn_words & ext_words) / max(len(syn_words), 1)
    resolution_score = min(30, int(resolution_overlap * 60))
    score += resolution_score
    if resolution_score < 10:
        gaps.append(
            "Answer did not reach synthesis — THEOS synthesis identified "
            "integrative conclusions not present in the submitted answer."
        )

    # --- Epistemic honesty (30 pts) ---
    if theos_halt_reason == "irreducible_uncertainty":
        # Check if the external answer acknowledged uncertainty
        uncertainty_words = {"uncertain", "unclear", "depends", "context",
                             "may", "might", "could", "debatable", "complex",
                             "nuanced", "tension", "however", "but", "although"}
        has_uncertainty = bool(uncertainty_words & set(_tokenize(external_answer)))
        epistemic_score = 30 if has_uncertainty else 10
        if not has_uncertainty:
            gaps.append(
                "THEOS found irreducible tension on this question — the answer "
                "should acknowledge genuine uncertainty rather than resolving "
                "a question that has no clean answer."
            )
    else:
        # No irreducible tension found — partial credit for consistency
        epistemic_score = 20 + int(theos_confidence * 10)
    score += min(30, epistemic_score)

    # Adjust for THEOS confidence
    final_score = min(100, int(score * (0.7 + 0.3 * theos_confidence)))

    # Grade
    if final_score >= 85:
        grade = "A"
        verdict = "Certified — answer demonstrates strong dialectical completeness."
    elif final_score >= 70:
        grade = "B"
        verdict = "Partially certified — good synthesis but missing adversarial depth."
    elif final_score >= 50:
        grade = "C"
        verdict = "Marginal — single-frame analysis; dialectical interrogation needed."
    elif final_score >= 30:
        grade = "D"
        verdict = "Uncertified — answer misses significant opposing considerations."
    else:
        grade = "F"
        verdict = "Uncertified — answer is single-perspective; run it through the wringer."

    return {
        "certification_score": final_score,
        "grade": grade,
        "verdict": verdict,
        "gaps": gaps,
        "subscores": {
            "dialectical_completeness": dialectical_score,
            "resolution_quality": resolution_score,
            "epistemic_honesty": min(30, epistemic_score),
        },
    }


# ============================================================
# REASONING COMMONS STORE
# ============================================================

class ReasoningCommons:
    """
    The shared dialectical wisdom layer across THEOS instances and
    THEOS-certified AI systems.

    Design principles:
    1. Quality-gated: only entries with phi > 0 and confidence > 0.6 admitted
    2. Cumulative: each new reasoning session is informed by past entries
    3. Open: any AI can query the commons before answering; any THEOS instance
       can contribute after running its wringer
    4. Federated: the commons endpoint accepts external contributions with
       source attribution — enabling the "Reasoning Commons" network vision

    Retrieval strategy:
    - Primary: TF-IDF cosine similarity on question text
    - Secondary: Phi trajectory similarity (prioritizes high-quality entries)
    - Combined score: 0.7 * semantic + 0.3 * trajectory_quality
    """

    # Minimum quality thresholds for admission
    MIN_CONFIDENCE = 0.60
    MIN_PHI = 0.01          # At least some genuine tension

    def __init__(self, store_path: str = "./commons_store.jsonl"):
        self.store_path = store_path
        self.entries: List[Dict[str, Any]] = []

        if os.path.exists(store_path):
            with open(store_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.entries.append(json.loads(line))
            print(f"ReasoningCommons: loaded {len(self.entries)} entries from {store_path}")
        else:
            print(f"ReasoningCommons: new store at {store_path}")

    def submit(
        self,
        question: str,
        constructive: str,
        adversarial: str,
        synthesis: str,
        phi_trajectory: List[float],
        halt_reason: str,
        confidence: float,
        phi_composite: float,
        passes_used: int,
        source: str = "theos_native",
        domain: str = "general",
        mode_used: int = 3,
        external_answer: Optional[str] = None,         # For certification entries
        certification_result: Optional[Dict] = None,    # From compute_certification_score
    ) -> str:
        """
        Submit a new entry to the Reasoning Commons.

        Quality gate: entries below MIN_CONFIDENCE or MIN_PHI are rejected.
        This ensures the Commons contains only entries where genuine dialectical
        tension was present and resolved (or honestly reported as unresolvable).

        Returns: entry_id (UUID) or empty string if rejected.
        """
        if confidence < self.MIN_CONFIDENCE:
            print(f"Commons: rejected (confidence {confidence:.2f} < {self.MIN_CONFIDENCE})")
            return ""
        if phi_composite < self.MIN_PHI:
            print(f"Commons: rejected (phi {phi_composite:.4f} < {self.MIN_PHI})")
            return ""

        # Fine-tune priority: high initial tension that resolved = most valuable
        if len(phi_trajectory) >= 2:
            initial_phi = phi_trajectory[0]
            final_phi = phi_trajectory[-1]
            resolution = max(0.0, initial_phi - final_phi)
            fine_tune_priority = min(1.0, resolution * confidence)
        else:
            fine_tune_priority = confidence * phi_composite

        entry_id = str(uuid.uuid4())
        entry = {
            "id": entry_id,
            "question": question,
            "constructive": constructive[:1500],
            "adversarial": adversarial[:1500],
            "synthesis": synthesis[:3000],
            "phi_trajectory": phi_trajectory,
            "halt_reason": halt_reason,
            "confidence": round(confidence, 4),
            "phi_composite": round(phi_composite, 4),
            "passes_used": passes_used,
            "domain": domain,
            "source": source,
            "mode_used": mode_used,
            "fine_tune_priority": round(fine_tune_priority, 4),
            "timestamp": datetime.now().isoformat(),
        }

        if external_answer is not None:
            entry["external_answer"] = external_answer[:1500]
        if certification_result is not None:
            entry["certification"] = certification_result

        self.entries.append(entry)

        with open(self.store_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        print(
            f"Commons: admitted '{question[:55]}...' "
            f"[phi={phi_composite:.3f}, conf={confidence:.2f}, "
            f"priority={fine_tune_priority:.2f}, source={source}]"
        )
        return entry_id

    def search(
        self,
        query: str,
        max_results: int = 5,
        domain: Optional[str] = None,
        min_confidence: float = 0.0,
        min_phi: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Search the Commons for relevant past reasoning.

        Returns entries sorted by combined score:
            0.7 * TF-IDF similarity + 0.3 * trajectory quality

        A high phi_trajectory (high fine_tune_priority) entry is worth
        more than a low-tension entry with the same semantic similarity.
        """
        if not self.entries:
            return []

        # Build corpus from combined text (question + synthesis) for richer TF-IDF
        corpus = [
            e["question"] + " " + e.get("synthesis", "")[:200]
            for e in self.entries
        ]
        scored = []

        for i, entry in enumerate(self.entries):
            if domain and entry.get("domain") != domain:
                continue
            if entry.get("confidence", 0) < min_confidence:
                continue
            if entry.get("phi_composite", 0) < min_phi:
                continue

            # Match against question + key content for better recall
            doc_text = (
                entry["question"] + " " +
                entry.get("synthesis", "")[:200] + " " +
                entry.get("adversarial", "")[:150]
            )
            semantic_sim = _tfidf_cosine(query, doc_text, corpus)
            quality_weight = entry.get("fine_tune_priority", 0.0)
            combined = 0.7 * semantic_sim + 0.3 * quality_weight

            if combined > 0.001:    # Skip irrelevant entries
                scored.append((combined, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:max_results]]

    def get_related_adversarial(self, query: str) -> Optional[str]:
        """
        Return the adversarial framing from the most relevant past entry.

        This is the key Commons benefit: before answering, an AI can look up
        what the adversarial engine previously said about similar questions.
        This seeds more complete reasoning without full wringer overhead.
        """
        results = self.search(query, max_results=1, min_phi=0.05)
        if results:
            return results[0].get("adversarial", "")
        return None

    def get_stats(self) -> Dict[str, Any]:
        if not self.entries:
            return {
                "total_entries": 0,
                "message": "Commons is empty — run queries through THEOS to populate it.",
            }

        avg_phi = sum(e.get("phi_composite", 0) for e in self.entries) / len(self.entries)
        avg_conf = sum(e.get("confidence", 0) for e in self.entries) / len(self.entries)
        avg_priority = sum(e.get("fine_tune_priority", 0) for e in self.entries) / len(self.entries)

        halt_counts: Counter = Counter(e.get("halt_reason", "unknown") for e in self.entries)
        sources: Counter = Counter(e.get("source", "unknown") for e in self.entries)
        domains: Counter = Counter(e.get("domain", "general") for e in self.entries)

        # Top 3 highest priority entries (best training material)
        top = sorted(self.entries, key=lambda x: x.get("fine_tune_priority", 0), reverse=True)[:3]
        top_preview = [
            {
                "question": e["question"][:80],
                "phi": e.get("phi_composite", 0),
                "priority": e.get("fine_tune_priority", 0),
                "halt_reason": e.get("halt_reason", "?"),
            }
            for e in top
        ]

        return {
            "total_entries": len(self.entries),
            "avg_phi_composite": round(avg_phi, 4),
            "avg_confidence": round(avg_conf, 4),
            "avg_fine_tune_priority": round(avg_priority, 4),
            "halt_reason_distribution": dict(halt_counts),
            "sources": dict(sources),
            "domains": dict(domains),
            "top_priority_entries": top_preview,
            "description": (
                "The Reasoning Commons accumulates dialectical wisdom across all "
                "THEOS instances. Each entry is quality-gated: genuine tension "
                "required (phi > 0.01) and confident resolution required (conf > 0.60). "
                "Fine-tune priority rewards high initial tension that resolved cleanly."
            ),
        }
