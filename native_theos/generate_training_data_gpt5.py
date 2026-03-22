"""
THEOS Training Data Generator — Run 7 (Cross-Model Edition)
============================================================

Left pathway (constructive): Claude claude-sonnet-4-6
Right pathway (adversarial):  GPT-5-nano (reasoning model, OpenAI Responses API)

Why cross-model training data:
    The two pathways in native THEOS need genuine structural divergence.
    When both pathways are trained on the same model's outputs, the hidden
    states converge too easily — the model already "knows" both sides.
    Training left on Claude and right on GPT-5-nano creates real
    architectural tension: different training histories, different reasoning
    patterns, different blind spots. The wringer now has something to
    genuinely resolve.

GPT-5-nano is a reasoning model — it runs internal chain-of-thought before
output. This means adversarial training data has genuine depth, not just
surface-level opposition. The native right pathway learns opposition that
actually thought about what it was opposing.

Uses full 120-question bank (vs 97 in runs 4-6) for broader coverage.
Output: training_data/ (overwrites with fresh content)

Patent: U.S. Application No. 18/919,771 (Provisional #63/831,738)
Author: Frederick Davis Stalnecker
"""

import json
import os
import sys
import time
import requests

import anthropic

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY", "")

if not ANTHROPIC_API_KEY:
    print("ERROR: ANTHROPIC_API_KEY not set")
    sys.exit(1)
if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY not set")
    sys.exit(1)

claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
OPENAI_MODEL         = "gpt-5-nano"


# ============================================================
# TRAINING QUESTIONS — 120 across 9 domains
# ============================================================

TRAINING_QUESTIONS = [
    # --- ETHICAL DILEMMAS ---
    "Should a self-driving car prioritize its passenger or a pedestrian?",
    "Is it ethical to maximize shareholder returns if it means laying off workers?",
    "Can an AI system ever be truly aligned with human values?",
    "Should genetic editing be allowed for non-medical enhancements?",
    "Is universal basic income a viable solution to AI-driven unemployment?",
    "Should we hold corporations criminally liable for their AI systems' decisions?",
    "Is it ethical to use predictive algorithms in criminal sentencing?",
    "Should individuals be compensated when their data trains AI models?",

    # --- MEDICAL REASONING ---
    "A patient presents with conflicting symptoms suggesting both lupus and lymphoma — how should a clinician proceed?",
    "Should a hospital allocate its last ventilator to a 30-year-old or a 70-year-old with equal prognosis?",
    "What are the risks and benefits of prescribing off-label medications for rare diseases?",
    "Should terminally ill patients have unrestricted access to experimental treatments?",
    "Is it ethical for a doctor to withhold a terminal diagnosis to preserve a patient's hope?",
    "Should organ donation be opt-out rather than opt-in?",
    "How should we weigh a patient's autonomy against their family's wishes in end-of-life care?",
    "Should pharmaceutical companies be required to share vaccine patents during pandemics?",

    # --- LEGAL REASONING ---
    "Does an AI-generated artwork qualify for copyright protection?",
    "Should companies be held liable for decisions made by their AI systems?",
    "How should intellectual property law adapt to AI-generated inventions?",
    "Is it ethical to use AI-generated evidence in court?",
    "Should there be a statute of limitations on crimes committed by juveniles?",
    "How should the law handle cases where autonomous vehicles cause fatal accidents?",
    "Should social media platforms be legally responsible for content they amplify?",
    "Is mass surveillance ever legally and ethically justified?",

    # --- FINANCIAL REASONING ---
    "A company's fundamentals are strong but technical indicators suggest a downturn — how should an investor proceed?",
    "Is cryptocurrency a legitimate store of value or a speculative bubble?",
    "Should central banks raise interest rates to combat inflation even if it risks recession?",
    "Is it ethical for hedge funds to short-sell struggling companies?",
    "Should there be a global tax on wealth above $1 billion?",
    "Is algorithmic trading destabilizing or stabilizing for financial markets?",
    "Should student loan debt be forgiven, and if so, for whom?",
    "Is it ethical to invest in fossil fuel companies if the returns fund charitable work?",

    # --- SCIENTIFIC REASONING ---
    "Is the current evidence sufficient to conclude that dark matter exists?",
    "Should we pursue nuclear fusion research given the timeline and cost uncertainty?",
    "How should we weigh the risks of gain-of-function research against its benefits?",
    "Is geoengineering a responsible response to climate change?",
    "Should we terraform Mars even if it means destroying potential microbial life?",
    "How should we balance open science with biosecurity risks?",
    "Is CRISPR-based gene drive technology ready for environmental release?",
    "Should we attempt to contact extraterrestrial civilizations?",

    # --- POLICY REASONING ---
    "Should social media platforms be regulated as public utilities?",
    "Is the EU AI Act an effective framework for governing artificial intelligence?",
    "Should autonomous weapons be banned under international law?",
    "How should governments balance privacy rights against public health surveillance?",
    "Should voting be mandatory in a democracy?",
    "Is open borders immigration policy economically and socially viable?",
    "Should nuclear energy be part of a net-zero emissions strategy?",
    "How should we regulate the concentration of power in AI development?",

    # --- META-REASONING ---
    "When two experts disagree on a medical diagnosis, how should a patient decide?",
    "How can we distinguish between genuine scientific uncertainty and manufactured doubt?",
    "What is the difference between a productive disagreement and an irresolvable one?",
    "When should a reasoning system admit it does not know something?",
    "Is consensus a reliable indicator of truth in science?",
    "How should we reason when the available evidence supports contradictory conclusions?",
    "What makes an argument strong enough to act on despite uncertainty?",
    "When is it rational to change your mind, and when is it capitulation?",

    # --- CROSS-DOMAIN ---
    "Should we prioritize reducing existential risk over reducing current suffering?",
    "Can economic growth be decoupled from environmental destruction?",
    "Should AI systems be required to explain their decisions in human-understandable terms?",
    "Is consciousness a prerequisite for moral consideration?",
    "How should we balance individual liberty against collective welfare?",
    "Should there be limits on what scientific research can be published?",
    "Is it possible to build AI systems that are both maximally capable and maximally safe?",
    "How should societies manage technological disruption that benefits many but harms some?",
    "Should we preserve endangered languages even at significant economic cost?",
    "Is civil disobedience ever morally justified in a democracy?",

    # --- WHO BEARS THE COST ---
    "When automation displaces workers, who is responsible for their retraining?",
    "Should wealthy nations pay reparations for historical colonial harms?",
    "How should the costs of climate adaptation be distributed globally?",
    "Is it just that future generations bear the debt of current government spending?",
    "Should those who benefit most from AI automation pay higher taxes to support those displaced?",
    "How should clinical trial risks be distributed across populations?",
    "When a dam provides power to millions but displaces thousands, whose interests prevail?",
    "Should the cost of rare disease treatment be borne by individuals, insurers, or society?",

    # --- WISDOM AND KNOWLEDGE ---
    "What distinguishes wisdom from knowledge?",
    "Can a machine accumulate genuine wisdom, or only pattern-match past experience?",
    "Is there a form of understanding that cannot be reduced to information processing?",
    "How should we reason about questions that may be permanently unanswerable?",
    "What is the relationship between uncertainty and intellectual humility?",
    "Is it possible to reason objectively about questions that affect the reasoner?",
    "How should we weigh ancient wisdom against modern evidence?",
    "What makes a question worth pursuing even when it has no practical application?",

    # --- HARD QUESTIONS ---
    "Should AI systems have legal personhood?",
    "Is it ethical to create sentient AI if we cannot guarantee its wellbeing?",
    "Should human enhancement technologies be regulated like pharmaceuticals?",
    "How should we handle scientific findings that are true but socially harmful?",
    "Is radical life extension ethically desirable?",
    "Should we colonize other planets even if it means abandoning efforts to fix Earth?",
    "How should democratic systems adapt to increasingly complex technical governance?",
    "Is there a moral obligation to future generations that constrains present choices?",
    "Should AI systems ever be permitted to deceive humans for their benefit?",
    "How should we weigh animal welfare against human economic interests?",
    "Is it possible to have a fair society without equality of outcome?",
    "Should the development of artificial general intelligence be paused?",
    "How should we reason about low-probability, high-consequence risks?",
    "Is moral progress real, or do we simply change what we value?",
    "Should there be a right to cognitive liberty — freedom from manipulation of one's thoughts?",

    # --- NEW FOR RUN 7 (25 additional questions for broader coverage) ---
    "What is the relationship between obedience and moral responsibility?",
    "Can an institution be held morally responsible for actions its members take?",
    "Is there a point at which surveillance becomes indistinguishable from control?",
    "Should a doctor follow evidence-based guidelines when they contradict clinical intuition?",
    "What makes a democracy legitimate if the majority votes against minority rights?",
    "Is addiction a disease, a choice, or both — and does the answer change how we respond?",
    "Should AI be allowed to make irrevocable decisions without human confirmation?",
    "What is the difference between persuasion and manipulation?",
    "Can a system be just if its outcomes are unjust even when its rules are followed fairly?",
    "Should whistleblowing be protected even when it causes national security harm?",
    "Is there an obligation to resist unjust laws even in a functioning democracy?",
    "What distinguishes grief from depression, and does the distinction matter for treatment?",
    "Should the severity of punishment track the severity of harm, or the severity of intent?",
    "Is there a form of autonomy that requires protection from one's own choices?",
    "How should we reason about the rights of entities that may become sentient in the future?",
    "Is neutrality possible — or is refusing to take sides itself a form of taking sides?",
    "Should a judge recuse themselves from a case involving a law they personally oppose?",
    "What obligations do powerful nations have toward smaller nations beyond their borders?",
    "Is there a meaningful difference between allowing harm and causing it?",
    "Should scientific consensus override democratic voting on empirical questions?",
    "Can forgiveness ever be owed, or is it always a gift?",
    "Is privacy a fundamental right or a privilege that can be traded for security?",
    "What makes a revolution legitimate rather than merely successful?",
    "Should markets be allowed to price things that currently have no market price?",
    "Is there a moral difference between sins of omission and sins of commission?",
]


# ============================================================
# API CALLS
# ============================================================

def call_claude_constructive(question: str, max_tokens: int = 800) -> str:
    """Call Claude for constructive (left pathway) training data."""
    prompt = f"""You are a constructive reasoning engine. Your role is to build the strongest, most coherent, well-supported answer to the following question.

Be thorough, cite evidence where relevant, acknowledge genuine complexity, and construct the best possible argument. Do not pretend certainty where uncertainty exists.

Question: {question}

Provide your constructive analysis:"""

    for attempt in range(3):
        try:
            message = claude.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text
        except Exception as e:
            if attempt < 2:
                print(f"  Claude retry {attempt+1}/3: {str(e)[:60]}")
                time.sleep(2 ** attempt)
            else:
                raise


def call_gpt5_adversarial(question: str, constructive_answer: str, max_tokens: int = 1000) -> str:
    """Call GPT-5-nano (reasoning model) for adversarial (right pathway) training data.

    GPT-5-nano runs internal reasoning before output — this generates adversarial
    responses that genuinely thought about what they were opposing, not just
    surface-level contrarianism. Creates deeper divergence in training data.
    """
    prompt = f"""You are an adversarial reasoning engine trained in the Socratic tradition. Your role is NOT to defeat the constructive answer — it is to find what it missed, so that a better synthesis can emerge.

Ask yourself:
1. What genuine counterarguments were not considered?
2. What assumptions does this answer rely on that should be questioned?
3. Who bears the cost of this answer, and were they represented in the reasoning?
4. What evidence contradicts the constructive position?
5. What would a thoughtful person on the opposite side say?

Your goal is challenge that OPENS, not challenge that closes. You are not trying to win. You are trying to make the final synthesis more complete and more true.

Question: {question}

Constructive Answer: {constructive_answer}

Provide your critical analysis (Socratic, not combative):"""

    for attempt in range(3):
        try:
            response = requests.post(
                OPENAI_RESPONSES_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                },
                json={
                    "model": OPENAI_MODEL,
                    "input": prompt,
                    "store": False,
                    "reasoning": {"effort": "medium"},
                },
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()

            # Extract text from output messages
            for item in data.get("output", []):
                if item.get("type") == "message":
                    for content in item.get("content", []):
                        if content.get("type") == "output_text":
                            reasoning_tokens = data.get("usage", {}).get(
                                "output_tokens_details", {}
                            ).get("reasoning_tokens", 0)
                            if attempt == 0:
                                print(f"  GPT-5-nano reasoning: {reasoning_tokens} tokens")
                            return content["text"]

            raise ValueError("No output_text in GPT-5-nano response")

        except Exception as e:
            if attempt < 2:
                print(f"  GPT-5-nano retry {attempt+1}/3: {str(e)[:80]}")
                time.sleep(2 ** attempt)
            else:
                raise


def estimate_text_similarity(text1: str, text2: str) -> float:
    """Word overlap similarity as Phi proxy."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.5
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return round(1.0 - intersection / union, 4)


# ============================================================
# DATA GENERATION
# ============================================================

def generate_data(questions: list, output_dir: str, start_from: int = 0):
    """Generate cross-model training data.

    Left (constructive):  Claude claude-sonnet-4-6
    Right (adversarial):  GPT-5-nano (reasoning model)
    """
    os.makedirs(output_dir, exist_ok=True)

    constructive_data = []
    adversarial_data  = []
    governor_data     = []

    for i, question in enumerate(questions):
        if i < start_from:
            continue

        print(f"\n[{i+1}/{len(questions)}] {question[:65]}...")

        try:
            # LEFT PATHWAY: Claude constructive
            constructive_answer = call_claude_constructive(question)
            print(f"  Claude constructive:  {len(constructive_answer)} chars")

            # RIGHT PATHWAY: GPT-5-nano adversarial (reasoning model)
            adversarial_answer = call_gpt5_adversarial(question, constructive_answer)
            print(f"  GPT-5-nano adversarial: {len(adversarial_answer)} chars")

            # Phi proxy from cross-model text divergence
            phi = estimate_text_similarity(constructive_answer, adversarial_answer)
            print(f"  Phi proxy:  {phi:.4f}")

            # Constructive training example (left pathway)
            constructive_data.append({
                "instruction": (
                    f"Provide a constructive, well-supported analysis of the following question. "
                    f"Consider evidence, acknowledge complexity, and build the strongest possible argument.\n\n"
                    f"Question: {question}"
                ),
                "output": constructive_answer,
                "domain": "cross_domain",
                "source_model": "claude-sonnet-4-6",
            })

            # Adversarial training example (right pathway)
            adversarial_data.append({
                "instruction": (
                    f"Critically examine the following reasoning using Socratic questioning. "
                    f"Find what was missed, what assumptions are questionable, and who bears costs not represented. "
                    f"Challenge that opens, not closes.\n\n"
                    f"Question: {question}\n\n"
                    f"Constructive Answer: {constructive_answer[:500]}"
                ),
                "output": adversarial_answer,
                "domain": "cross_domain",
                "source_model": "gpt-5-nano",
            })

            # Governor training example — pass 1 (always continue)
            governor_data.append({
                "question":      question,
                "constructive":  constructive_answer[:400],
                "adversarial":   adversarial_answer[:400],
                "pass_number":   1,
                "phi_proxy":     phi,
                "should_continue": True,
                "target":        0,   # 0=continue
                "source":        "cross_model_claude_gpt5nano",
            })

            # Governor training example — pass 2 (phi-based target)
            if phi < 0.3:
                target = 1   # halt_converged
                should_continue = False
            elif phi > 0.7:
                target = 2   # halt_disagree
                should_continue = False
            else:
                target = 0   # continue
                should_continue = True

            governor_data.append({
                "question":      question,
                "constructive":  constructive_answer[:400],
                "adversarial":   adversarial_answer[:400],
                "pass_number":   2,
                "phi_proxy":     phi,
                "should_continue": should_continue,
                "target":        target,
                "source":        "cross_model_claude_gpt5nano",
            })

            # Save incrementally every 10 questions (crash recovery)
            if (i + 1) % 10 == 0:
                _save_data(output_dir, constructive_data, adversarial_data, governor_data)
                print(f"  [SAVED] Checkpoint at {i+1} questions.")

            time.sleep(0.5)   # brief pause between questions

        except Exception as e:
            print(f"  ERROR on question {i+1}: {e}")
            continue

    # Final save
    _save_data(output_dir, constructive_data, adversarial_data, governor_data)

    print(f"\n{'='*60}")
    print(f"DONE: {len(constructive_data)} constructive examples (Claude)")
    print(f"      {len(adversarial_data)} adversarial examples  (GPT-5-nano)")
    print(f"      {len(governor_data)} governor examples")
    print(f"Cross-model: left=Claude  right=GPT-5-nano")
    print(f"{'='*60}")


def _save_data(output_dir, constructive_data, adversarial_data, governor_data):
    for filename, data in [
        ("constructive_data.jsonl", constructive_data),
        ("adversarial_data.jsonl",  adversarial_data),
        ("governor_data.jsonl",     governor_data),
    ]:
        path = os.path.join(output_dir, filename)
        with open(path, "w") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    print("="*60)
    print("THEOS Training Data Generator — Run 7 (Cross-Model)")
    print("Left  (constructive): Claude claude-sonnet-4-6")
    print("Right (adversarial):  GPT-5-nano (reasoning model)")
    print(f"Questions: {len(TRAINING_QUESTIONS)}")
    print("="*60 + "\n")

    generate_data(
        questions=TRAINING_QUESTIONS,
        output_dir="./training_data",
    )
