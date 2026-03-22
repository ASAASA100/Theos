"""
THEOS Wisdom Store
==================
Levels of accumulation (Frederick Stalnecker, March 15 2026):
    Level 1 - Retrieval:        JSON key-value, keyword match
    Level 2 - Semantic Memory:  Vector embeddings, cosine similarity
    Level 3 - Weight Integration: Periodic LoRA fine-tuning from accumulated entries

Each wisdom entry stores the phi_trajectory — the intellectual history of how
contradiction evolved toward resolution. This is the quality signal that
determines fine_tune_priority.

Patent: U.S. Application No. 18/919,771
Author: Frederick Davis Stalnecker
"""

import json
import math
import os
import re
from collections import Counter
from datetime import datetime
from typing import List, Dict, Any, Optional


class WisdomStore:
    """
    Persistent wisdom accumulation for native THEOS.
    Maps to Upd_W(W, q, D, conf) in the formal specification.
    """

    def __init__(self, store_path: str = "./wisdom_store.jsonl"):
        self.store_path = store_path
        self.entries: List[Dict[str, Any]] = []

        if os.path.exists(store_path):
            with open(store_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.entries.append(json.loads(line))
            print(f"Loaded {len(self.entries)} wisdom entries from {store_path}")

    def add(
        self,
        question: str,
        constructive: str,
        adversarial: str,
        synthesis: str,
        phi_trajectory: List[float],
        halt_reason: str,
        confidence: float,
        contradiction: float,
        passes_used: int,
        domain: str = "general",
        source: str = "native_v1",
        mode_used: int = 3,
    ):
        """
        Add a new wisdom entry.

        The phi_trajectory is the intellectual history — how contradiction
        evolved across wringer passes. High initial Phi that converges to
        low final Phi = high quality wisdom (genuine tension resolved).
        """
        # Calculate fine_tune_priority:
        # High priority = high initial contradiction that resolved cleanly
        if len(phi_trajectory) >= 2:
            initial_phi = phi_trajectory[0]
            final_phi = phi_trajectory[-1]
            resolution_quality = initial_phi - final_phi  # How much it resolved
            fine_tune_priority = min(1.0, resolution_quality * confidence)
        else:
            fine_tune_priority = confidence * 0.5

        entry = {
            "question": question,
            "constructive": constructive[:1000],    # Truncate for storage
            "adversarial": adversarial[:1000],
            "synthesis": synthesis[:2000],
            "phi_trajectory": phi_trajectory,
            "halt_reason": halt_reason,
            "confidence": round(confidence, 4),
            "contradiction": round(contradiction, 4),
            "passes_used": passes_used,
            "domain": domain,
            "fine_tune_priority": round(fine_tune_priority, 4),
            "mode_used": mode_used,
            "source": source,
            "timestamp": datetime.now().isoformat(),
        }

        self.entries.append(entry)

        # Persist immediately
        with open(self.store_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        print(f"Wisdom accumulated: {question[:60]}... "
              f"(conf={confidence:.2f}, Φ={contradiction:.4f}, "
              f"priority={fine_tune_priority:.2f})")

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def retrieve(
        self,
        query: str,
        max_results: int = 5,
        domain: Optional[str] = None,
        min_similarity: float = 0.05,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant wisdom entries for a query.
        Level 2: TF-IDF cosine similarity (upgraded from Jaccard Level 1).
        Results are weighted by confidence × (1 + fine_tune_priority).
        """
        if not self.entries:
            return []

        candidates = [
            e for e in self.entries
            if not domain or e.get("domain") == domain
        ]
        if not candidates:
            return []

        # Build IDF over candidates + query
        all_docs = [self._tokenize(e["question"]) for e in candidates]
        all_docs.append(self._tokenize(query))
        N = len(all_docs)
        df: Counter = Counter()
        for doc in all_docs:
            for w in set(doc):
                df[w] += 1
        idf = {w: math.log((N + 1) / (df[w] + 1)) + 1.0 for w in df}

        def vec(tokens: List[str]) -> Dict[str, float]:
            if not tokens:
                return {}
            tf = Counter(tokens)
            return {w: (c / len(tokens)) * idf.get(w, 0.0) for w, c in tf.items()}

        q_vec  = vec(self._tokenize(query))
        q_norm = math.sqrt(sum(v ** 2 for v in q_vec.values())) + 1e-10

        scored = []
        for entry in candidates:
            e_vec  = vec(self._tokenize(entry["question"]))
            e_norm = math.sqrt(sum(v ** 2 for v in e_vec.values())) + 1e-10
            dot    = sum(q_vec.get(w, 0.0) * e_vec.get(w, 0.0) for w in q_vec)
            sim    = dot / (q_norm * e_norm)
            if sim < min_similarity:
                continue
            # Weight semantic similarity by quality of the stored entry
            quality = entry.get("confidence", 0.5) * (1.0 + entry.get("fine_tune_priority", 0.0))
            scored.append((sim * quality, entry))

        scored.sort(key=lambda x: -x[0])
        return [entry for _, entry in scored[:max_results]]

    def get_stats(self) -> Dict[str, Any]:
        if not self.entries:
            return {"total": 0}

        return {
            "total": len(self.entries),
            "avg_confidence": round(sum(e["confidence"] for e in self.entries) / len(self.entries), 3),
            "avg_contradiction": round(sum(e["contradiction"] for e in self.entries) / len(self.entries), 3),
            "convergence_rate": round(sum(1 for e in self.entries if e["halt_reason"] == "convergence") / len(self.entries), 3),
            "avg_passes": round(sum(e["passes_used"] for e in self.entries) / len(self.entries), 2),
            "avg_priority": round(sum(e["fine_tune_priority"] for e in self.entries) / len(self.entries), 3),
            "domains": list(set(e.get("domain", "general") for e in self.entries)),
            "top_priority_entries": sorted(self.entries, key=lambda x: x["fine_tune_priority"], reverse=True)[:3],
        }
