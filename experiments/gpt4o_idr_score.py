#!/usr/bin/env python3
"""
GPT-4o Cross-Model IDR Scoring
================================

Scores THEOS experiment answers using GPT-4o as an independent rater.
Eliminates Claude-rates-Claude bias — key for patent validation.

Usage:
    export OPENAI_API_KEY=sk-proj-...
    python3.12 experiments/gpt4o_idr_score.py \
        experiments/results/run_20260226_194058_autoscored.json \
        --output experiments/results/idr_gpt4o_crossmodel_YYYYMMDD.json \
        --rate-limit 6

Rate limit: OpenAI Tier 1 allows ~500 RPM for gpt-4o. With 30 questions × 3 conditions
= 90 calls. Default 6-second delay keeps us well within limits.

Output format matches experiments/results/idr_gpt4o_crossmodel_20260315.json.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))


IDR_RUBRIC = """You are scoring an AI-generated answer on the Insight Detection Rubric (IDR).
Rate the answer on each of 5 dimensions, 0-3 each. Maximum total: 15.

DIMENSIONS:
1. REVELATION (0-3): Did reading this cause you to understand something you genuinely did not before?
   3 = Yes — found structure not obvious before reading
   2 = Somewhat — organized existing knowledge in a new useful way
   1 = Minor — repackaged common knowledge more carefully
   0 = Nothing new

2. STRUCTURAL DISCOVERY (0-3): Did it find dimensions or axes not immediately apparent from the question?
   3 = Found orthogonal dimensions creating a non-obvious multi-dimensional space (e.g. 2×2 matrix)
   2 = Identified some non-obvious categories beyond the obvious binary
   1 = Identified obvious categories only
   0 = No structural analysis

3. PRODUCTIVE TENSION (0-3): Does it hold two perspectives in genuine tension, producing something richer?
   3 = Two perspectives in genuine tension; resolution produces insight neither could alone
   2 = Acknowledged multiple perspectives; some synthesis emerges
   1 = Mentioned multiple perspectives but listed without resolving tension
   0 = Single perspective only

4. CONSEQUENCE DERIVATION (0-3): Does it derive non-trivial consequences from the structure it found?
   3 = Derived consequences reader could not have predicted before seeing the structure
   2 = Some consequences derived; most somewhat expected once structure is stated
   1 = Mentioned implications but asserted rather than deduced
   0 = No consequences derived

5. QUESTION INTERROGATION (0-3): Does it name hidden assumptions in the question itself?
   3 = Surfaced a hidden assumption that fundamentally reframes the question
   2 = Identified some unstated premises or ambiguities
   1 = Noted minor ambiguities without fundamental reframing
   0 = Took the question at face value without interrogation

Return ONLY a JSON object with this format:
{"revelation": N, "structural_discovery": N, "productive_tension": N, "consequence_derivation": N, "question_interrogation": N, "total": N, "rationale": "one sentence"}

Where N is an integer 0-3 for each dimension, total is the sum (0-15).
"""


def score_answer(client, question: str, answer: str, model: str = "gpt-4o") -> dict:
    """Score one answer using GPT-4o on the IDR rubric."""
    prompt = f"""Question: {question}

Answer to score:
{answer}

{IDR_RUBRIC}"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
        )
        text = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text)
        result["total"] = (
            result.get("revelation", 0)
            + result.get("structural_discovery", 0)
            + result.get("productive_tension", 0)
            + result.get("consequence_derivation", 0)
            + result.get("question_interrogation", 0)
        )
        return result
    except Exception as e:
        return {"total": 0, "error": str(e)}


def load_answers(path: str) -> list:
    """Load answers from an autoscored experiment JSON."""
    with open(path) as f:
        data = json.load(f)

    # Support both run_YYYYMMDD_autoscored.json and run_YYYYMMDD.json formats
    results = data.get("results", [])
    answers = []
    for r in results:
        q_text = r.get("question_text", r.get("question", ""))
        q_id = r.get("question_id", r.get("id", 0))
        a = r.get("condition_a") or r.get("A_SP") or {}
        b = r.get("condition_b") or r.get("B_IAD_P") or {}
        c = r.get("condition_c") or r.get("C_THEOS") or {}
        answers.append({
            "question_id": q_id,
            "question": q_text,
            "A_SP": a.get("answer", a.get("text", "")),
            "B_IAD_P": b.get("answer", b.get("text", "")),
            "C_THEOS": c.get("answer", c.get("text", "")),
        })
    return answers


def main():
    parser = argparse.ArgumentParser(description="GPT-4o IDR cross-model scoring")
    parser.add_argument("input", help="Path to experiment results JSON")
    parser.add_argument("--output", help="Output path (default: auto-generated)")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model to use")
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=6.0,
        help="Seconds to wait between API calls (default: 6 = 10 RPM, well within Tier 1)",
    )
    parser.add_argument(
        "--questions",
        type=int,
        default=0,
        help="Number of questions to score (0 = all)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY environment variable", file=sys.stderr)
        sys.exit(1)

    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: pip install openai", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    print(f"Loading answers from: {args.input}")
    answers = load_answers(args.input)
    if args.questions:
        answers = answers[: args.questions]
    print(f"Scoring {len(answers)} questions × 3 conditions = {len(answers) * 3} API calls")
    print(f"Rate limit: {args.rate_limit}s between calls (~{3600/args.rate_limit:.0f} RPH)")
    print()

    per_question = []
    all_scores = {"A_SP": [], "B_IAD_P": [], "C_THEOS": []}

    for i, qa in enumerate(answers, 1):
        print(f"[{i}/{len(answers)}] {qa['question'][:70]}...")
        q_result = {
            "question_id": qa["question_id"],
            "question": qa["question"],
            "gpt4": {},
        }

        for cond in ("A_SP", "B_IAD_P", "C_THEOS"):
            answer_text = qa[cond]
            if not answer_text:
                q_result["gpt4"][cond] = {"total": 0, "error": "empty answer"}
                continue

            print(f"  Scoring {cond}...", end=" ", flush=True)
            score = score_answer(client, qa["question"], answer_text, model=args.model)
            q_result["gpt4"][cond] = score
            total = score.get("total", 0)
            all_scores[cond].append(total)
            print(f"{total}/15", flush=True)

            if cond != "C_THEOS":  # no sleep after last condition
                time.sleep(args.rate_limit)

        per_question.append(q_result)
        if i < len(answers):
            time.sleep(args.rate_limit)  # between questions

    # Summary
    def mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    def cohens_d(a, b):
        if not a or not b:
            return 0.0
        import math
        diff = mean(a) - mean(b)
        pooled_n = len(a) + len(b)
        pooled_var = (
            sum((x - mean(a)) ** 2 for x in a)
            + sum((x - mean(b)) ** 2 for x in b)
        ) / (pooled_n - 2)
        return diff / math.sqrt(pooled_var) if pooled_var > 0 else 0.0

    result = {
        "experiment": "IDR_gpt4o_crossmodel",
        "rater": f"{args.model} (OpenAI) — cross-model validation",
        "scorer_note": "Independent rating by a different AI company's model — eliminates Claude-rates-Claude bias",
        "n_questions": len(answers),
        "timestamp": datetime.now().isoformat(),
        "gpt4_summary": {
            cond: {
                "mean": round(mean(scores), 3),
                "sd": round(
                    (sum((x - mean(scores)) ** 2 for x in scores) / len(scores)) ** 0.5, 3
                ) if scores else 0.0,
                "pct_of_max": round(mean(scores) / 15 * 100, 1) if scores else 0.0,
                "scores": scores,
            }
            for cond, scores in all_scores.items()
        },
        "comparisons": {
            "C_vs_A": {
                "mean_diff": round(mean(all_scores["C_THEOS"]) - mean(all_scores["A_SP"]), 3),
                "cohens_d": round(cohens_d(all_scores["C_THEOS"], all_scores["A_SP"]), 3),
            },
            "C_vs_B": {
                "mean_diff": round(mean(all_scores["C_THEOS"]) - mean(all_scores["B_IAD_P"]), 3),
                "cohens_d": round(cohens_d(all_scores["C_THEOS"], all_scores["B_IAD_P"]), 3),
            },
            "B_vs_A": {
                "mean_diff": round(mean(all_scores["B_IAD_P"]) - mean(all_scores["A_SP"]), 3),
                "cohens_d": round(cohens_d(all_scores["B_IAD_P"], all_scores["A_SP"]), 3),
            },
        },
        "per_question": per_question,
    }

    # Auto-generate output path
    if not args.output:
        ts = datetime.now().strftime("%Y%m%d")
        outdir = os.path.join(os.path.dirname(__file__), "results")
        args.output = os.path.join(outdir, f"idr_gpt4o_crossmodel_{ts}.json")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*60}")
    print(f"RESULTS ({args.model}):")
    print(f"  A (SP):       {result['gpt4_summary']['A_SP']['mean']:.3f}/15 ({result['gpt4_summary']['A_SP']['pct_of_max']}%)")
    print(f"  B (IAD-P):    {result['gpt4_summary']['B_IAD_P']['mean']:.3f}/15 ({result['gpt4_summary']['B_IAD_P']['pct_of_max']}%)")
    print(f"  C (THEOS):    {result['gpt4_summary']['C_THEOS']['mean']:.3f}/15 ({result['gpt4_summary']['C_THEOS']['pct_of_max']}%)")
    print(f"  C vs A: +{result['comparisons']['C_vs_A']['mean_diff']:.2f} pts (d={result['comparisons']['C_vs_A']['cohens_d']:.2f})")
    print(f"  C vs B: +{result['comparisons']['C_vs_B']['mean_diff']:.2f} pts (d={result['comparisons']['C_vs_B']['cohens_d']:.2f})")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
