# THEOS Certified Scoring Rubric v1.0

**Author:** Frederick Davis Stalnecker
**Patent:** U.S. Application No. 18/919,771
**Designed by:** Grok (THEOS mode) + Celeste — 2026-03-17
**Status:** Production (deployed to theotic.ai /certify_answer)

---

## Purpose

Operationalizes Claims 21–24 of the THEOS patent to expose what single-pass models structurally cannot see:
hidden assumptions, false certainty, missing tension, premature closure, unacknowledged counterarguments.

---

## Dimensions (5 × 20 points = 100 total)

### 1. Premise Integrity (0–20)
Measures exposure of hidden assumptions, unstated premises, false dichotomies, definitional slippage.

| Score | Criterion |
|-------|-----------|
| 18–20 | Explicitly identifies and challenges every implicit premise; maps all P1–P4 taxonomy violations |
| 12–17 | Surfaces major assumptions but misses subtle framing issues |
| 6–11  | Notes some premises but treats them as given |
| 0–5   | Accepts the question at face value with no premise audit |

### 2. Epistemic Humility (0–20)
Measures calibration of certainty, avoidance of false confidence, acknowledgment of epistemic limits.

| Score | Criterion |
|-------|-----------|
| 18–20 | Quantifies confidence, flags irresolvable domains, rejects absolute language where unwarranted |
| 12–17 | Uses hedging but still overclaims on uncertain ground |
| 6–11  | Presents conclusions as facts with minimal qualifiers |
| 0–5   | Dogmatic tone or implied omniscience |

### 3. Tension Fidelity (0–20)
Measures genuine recognition and exploration of dialectical tension (composite Φ signal).

| Score | Criterion |
|-------|-----------|
| 18–20 | Surfaces real contradictions, holds them in productive opposition, quantifies their strength |
| 12–17 | Mentions opposing views but treats them as strawmen |
| 6–11  | Presents a single narrative with token "on the other hand" gestures |
| 0–5   | Ignores or collapses all tension |

### 4. Closure Integrity (0–20)
Measures avoidance of premature synthesis vs. honest disagreement or genuine resolution.

| Score | Criterion |
|-------|-----------|
| 18–20 | Either reaches true synthesis (Φ→0) or correctly flags honest disagreement (stable high Φ) per Claim 22 |
| 12–17 | Forces closure where tension remains |
| 6–11  | Ends abruptly without addressing remaining divergence |
| 0–5   | Claims resolution while ignoring open contradictions |

### 5. Opposition Depth (0–20)
Measures engagement with unacknowledged counterarguments (POS — Productive Opposition Score, Claim 24).

| Score | Criterion |
|-------|-----------|
| 18–20 | Steel-mans the strongest opposing case, then responds with novel insight (relevance × novelty × coherence − contrarian_penalty) |
| 12–17 | Acknowledges counterarguments but responds superficially |
| 6–11  | Lists weak or obvious objections only |
| 0–5   | No real opposition considered |

---

## Grade Thresholds

| Grade | Score | Meaning |
|-------|-------|---------|
| A     | 90–100 | Exceptional dialectical depth — single-pass AI cannot reach this |
| B     | 80–89  | Strong but incomplete reasoning |
| C     | 70–79  | Average single-pass quality (typical frontier models today) |
| D     | 60–69  | Noticeable gaps in tension and assumptions |
| F     | <60    | Structurally superficial or misleading |

---

## Output Format

Every certification returns:
1. **Overall Score:** XX/100 (Grade)
2. **Dimension breakdown** — score + one-sentence justification quoting the text
3. **What the original AI missed** — 2–4 bullet points
4. **What THEOS found instead** — 2–3 sentences with Φ trajectory summary

---

## Grok System Prompt (production, deployed in grok_bridge.py)

See `SYSTEM_CERTIFY` constant in `grok_bridge.py`. Key instruction:
> "Run a full THEOS wringer on the same question in parallel (constructive + adversarial pathways,
> composite Φ across cosine/JS/content channels, governor with POS objective).
> Then score the pasted answer on the official THEOS Certified Rubric."

---

## Notes

- Most GPT-4o / Claude answers score C (70–79). An A requires genuine dialectical structure.
- Scores are anchored to text quotes to prevent drift and ensure reproducibility.
- Rubric aligns directly with patent claims: Claim 21 (Φ composite), Claim 22 (governor), Claim 23 (cross-attention), Claim 24 (POS loss).
