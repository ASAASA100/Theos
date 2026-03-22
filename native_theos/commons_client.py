"""
THEOS Reasoning Commons Client
================================

A lightweight Python client for any AI system to access the Reasoning Commons.

Usage:
    from commons_client import CommonsClient

    client = CommonsClient("http://104.131.169.165:8000")

    # Look up past THEOS reasoning before answering
    context = client.lookup("Should AI systems be given legal personhood?")
    if context:
        print("THEOS previously found adversarial framing:")
        print(context["adversarial_framing"])

    # Submit your answer for THEOS certification
    cert = client.certify(
        question="Should AI systems be given legal personhood?",
        answer="AI systems should be treated as legal persons because...",
        source="my_ai_system_v1",
    )
    print(f"Certification: {cert['grade']} ({cert['certification_score']}/100)")
    print(f"Verdict: {cert['verdict']}")
    for gap in cert['gaps']:
        print(f"  Gap: {gap}")

Author: Frederick Davis Stalnecker
Patent: U.S. Application No. 18/919,771 (Provisional #63/831,738)
"""

import json
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError


class CommonsClient:
    """
    Client for the THEOS Reasoning Commons.

    No external dependencies — uses Python stdlib urllib only.
    Works with any Python 3.8+ environment.

    The Reasoning Commons is a quality-gated shared wisdom pool:
    - Search it before answering to see what THEOS found on similar questions
    - Submit answers for THEOS certification to get a structured score
    - Contribute your own THEOS results to grow the commons
    """

    def __init__(self, base_url: str = "http://104.131.169.165:8000", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _get(self, path: str, params: Optional[Dict] = None) -> Dict:
        url = f"{self.base_url}{path}"
        if params:
            from urllib.parse import urlencode
            url = f"{url}?{urlencode(params)}"
        try:
            with urlopen(url, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except URLError as e:
            raise ConnectionError(f"Commons unreachable at {url}: {e}")

    def _post(self, path: str, data: Dict) -> Dict:
        url = f"{self.base_url}{path}"
        payload = json.dumps(data).encode()
        req = Request(url, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except URLError as e:
            raise ConnectionError(f"Commons unreachable at {url}: {e}")

    # ----------------------------------------------------------------
    # Core methods
    # ----------------------------------------------------------------

    def lookup(
        self,
        question: str,
        max_results: int = 3,
        min_phi: float = 0.05,
    ) -> Optional[Dict[str, Any]]:
        """
        Look up the best matching past THEOS reasoning for a question.

        Returns the top result or None if nothing relevant found.
        Use this before answering to seed more complete reasoning.

        The returned dict includes:
            question      — the original question THEOS reasoned about
            synthesis     — THEOS's final synthesis
            adversarial_framing — what the adversarial engine challenged
            phi_composite — how much genuine tension was present
            confidence    — THEOS's confidence in the resolution
        """
        results = self.search(question, max_results=max_results, min_phi=min_phi)
        return results[0] if results else None

    def search(
        self,
        question: str,
        max_results: int = 5,
        domain: Optional[str] = None,
        min_phi: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Search the Commons for past reasoning on similar questions.

        Returns a list of matching entries, ranked by relevance and quality.
        """
        params = {"q": question, "max_results": max_results, "min_phi": min_phi}
        if domain:
            params["domain"] = domain
        resp = self._get("/commons/search", params)
        return resp.get("results", [])

    def certify(
        self,
        question: str,
        answer: str,
        source: str = "external_ai",
    ) -> Dict[str, Any]:
        """
        Submit an answer for THEOS certification.

        THEOS runs its full wringer on the question and scores your answer:
            certification_score  0-100
            grade                A/B/C/D/F
            verdict              One-sentence summary
            gaps                 List of specific gaps found
            subscores            {dialectical_completeness, resolution_quality,
                                  epistemic_honesty}
            theos_synthesis      What THEOS concluded
            theos_phi_score      How much genuine tension exists on this question

        Requires the THEOS server to have a loaded model.
        Raises HTTPError 503 if model not loaded.
        """
        return self._post("/certify", {
            "question": question,
            "answer_to_certify": answer,
            "source": source,
        })

    def contribute(
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
        source: str = "external_theos",
        domain: str = "general",
    ) -> str:
        """
        Contribute a THEOS result from your own instance to the shared Commons.

        Entries are quality-gated: confidence >= 0.60 and phi >= 0.01 required.
        Returns the entry_id if admitted, raises ValueError if rejected.
        """
        resp = self._post("/commons/submit", {
            "question": question,
            "constructive": constructive,
            "adversarial": adversarial,
            "synthesis": synthesis,
            "phi_trajectory": phi_trajectory,
            "halt_reason": halt_reason,
            "confidence": confidence,
            "phi_composite": phi_composite,
            "passes_used": passes_used,
            "source": source,
            "domain": domain,
        })
        if resp.get("status") == "admitted":
            return resp["id"]
        raise ValueError(f"Commons rejected entry: {resp}")

    def stats(self) -> Dict[str, Any]:
        """Return Commons statistics: entry count, quality metrics, sources."""
        return self._get("/commons/stats")

    def health(self) -> Dict[str, Any]:
        """Check if the THEOS server is healthy and model is loaded."""
        return self._get("/health")

    # ----------------------------------------------------------------
    # Convenience: pre-answer enrichment
    # ----------------------------------------------------------------

    def enrich_prompt(self, question: str) -> str:
        """
        Generate an enriched prompt by prepending relevant Commons context.

        Returns a prompt string that includes:
        - The original question
        - The adversarial framing THEOS previously found
        - A note to consider the opposing perspective

        Usage:
            enriched = client.enrich_prompt("Is consciousness necessary for moral consideration?")
            # Pass `enriched` to your LLM instead of the raw question
        """
        result = self.lookup(question, min_phi=0.05)
        if not result:
            return question

        adv = result.get("adversarial_framing", "")
        phi = result.get("phi_composite", 0)
        synthesis_snippet = result.get("synthesis", "")[:200]

        return (
            f"{question}\n\n"
            f"[THEOS Commons context — phi={phi:.3f}]\n"
            f"Previous adversarial framing: {adv[:300]}\n"
            f"Previous synthesis direction: {synthesis_snippet}...\n"
            f"Consider the opposing perspective before concluding."
        )
