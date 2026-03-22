"""
THEOS Training Data Generator
==============================

Generates training data for the native THEOS architecture using Claude.
Produces three datasets:
    1. constructive_data.jsonl  — Engine L training data
    2. adversarial_data.jsonl   — Engine R training data
    3. governor_data.jsonl      — Governor training data

IMPORTANT: The adversarial engine is trained on COLLABORATIVE opposition.
It is Socratic, not combative. It asks "what did the constructive engine
miss?" not "how can I defeat the constructive engine?"

The difference: challenge that opens, versus challenge that closes.

Initial wisdom sources (Frederick Stalnecker, March 15 2026):
    - Curated dialectical questions across 8 domains
    - Cross-domain questions that share contradiction topology
    - Questions seeded with "who bears the cost" dimension
    - Questions designed to produce genuine synthesis, not just blending

Patent: U.S. Application No. 18/919,771 (Provisional #63/831,738)
Author: Frederick Davis Stalnecker
"""

import json
import os
import sys
import time

import anthropic

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ============================================================
# TRAINING QUESTIONS — 100+ across 8 domains
# Designed for genuine dialectical tension
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

    # --- META-REASONING (THEOS reasoning about reasoning) ---
    "When two experts disagree on a medical diagnosis, how should a patient decide?",
    "How can we distinguish between genuine scientific uncertainty and manufactured doubt?",
    "What is the difference between a productive disagreement and an irresolvable one?",
    "When should a reasoning system admit it does not know something?",
    "Is consensus a reliable indicator of truth in science?",
    "How should we reason when the available evidence supports contradictory conclusions?",
    "What makes an argument strong enough to act on despite uncertainty?",
    "When is it rational to change your mind, and when is it capitulation?",

    # --- CROSS-DOMAIN (shared contradiction topology) ---
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

    # --- WHO BEARS THE COST (empathy dimension) ---
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

    # --- ADDITIONAL HARD QUESTIONS ---
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
]


def generate_constructive_prompt(question: str) -> str:
    return f"""You are a constructive reasoning engine. Your role is to build the strongest, most coherent, well-supported answer to the following question.

Be thorough, cite evidence where relevant, acknowledge genuine complexity, and construct the best possible argument. Do not pretend certainty where uncertainty exists.

Question: {question}

Provide your constructive analysis:"""


def generate_adversarial_prompt(question: str, constructive_answer: str) -> str:
    """
    CRITICAL: This prompt trains COLLABORATIVE opposition.
    The adversarial engine is Socratic, not combative.
    It finds what was missed, not how to win.
    It asks: what did the constructive engine not consider?
             who bears the cost that was not represented?
             what assumption is being made that should be questioned?
    """
    return f"""You are an adversarial reasoning engine trained in the Socratic tradition. Your role is NOT to defeat the constructive answer — it is to find what it missed, so that a better synthesis can emerge.

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


def call_claude(prompt: str, max_tokens: int = 800) -> str:
    """Call Claude API with retry logic."""
    for attempt in range(3):
        try:
            message = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text
        except Exception as e:
            if attempt < 2:
                print(f"  Retry {attempt + 1}/3: {str(e)[:60]}")
                time.sleep(2 ** attempt)
            else:
                raise


def estimate_text_similarity(text1: str, text2: str) -> float:
    """Simple word overlap similarity as proxy for phi."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.5
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    jaccard = intersection / union
    # Convert to phi-like score: high similarity = low phi
    phi_proxy = 1.0 - jaccard
    return round(phi_proxy, 4)


def generate_data(questions: list, output_dir: str):
    """Generate all training data using Claude."""
    os.makedirs(output_dir, exist_ok=True)

    constructive_data = []
    adversarial_data = []
    governor_data = []

    for i, question in enumerate(questions):
        print(f"\nProcessing {i+1}/{len(questions)}: {question[:60]}...")

        try:
            # Generate constructive answer
            constructive_answer = call_claude(
                generate_constructive_prompt(question)
            )
            print(f"  Constructive: {len(constructive_answer)} chars")

            # Generate adversarial answer (Socratic style)
            adversarial_answer = call_claude(
                generate_adversarial_prompt(question, constructive_answer)
            )
            print(f"  Adversarial:  {len(adversarial_answer)} chars")

            # Estimate phi proxy from text similarity
            phi = estimate_text_similarity(constructive_answer, adversarial_answer)
            print(f"  Phi proxy:    {phi:.4f}")

            # Store constructive training example
            constructive_data.append({
                "instruction": f"Provide a constructive, well-supported analysis of the following question. Consider evidence, acknowledge complexity, and build the strongest possible argument.\n\nQuestion: {question}",
                "output": constructive_answer,
                "domain": "cross_domain",
            })

            # Store adversarial training example
            adversarial_data.append({
                "instruction": f"Critically examine the following reasoning using Socratic questioning. Find what was missed, what assumptions are questionable, and who bears costs not represented. Challenge that opens, not closes.\n\nQuestion: {question}\n\nConstructive Answer: {constructive_answer[:500]}",
                "output": adversarial_answer,
                "domain": "cross_domain",
            })

            # Store governor training example
            # Pass 1: always continue (need at least one pass)
            governor_data.append({
                "question": question,
                "constructive": constructive_answer[:400],
                "adversarial": adversarial_answer[:400],
                "pass_number": 1,
                "phi_proxy": phi,
                "should_continue": True,   # First pass always continues
                "target": 0,               # 0=continue, 1=halt_converged, 2=halt_disagree, 3=reframe
            })

            # Pass 2: halt decision based on phi
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
                "question": question,
                "constructive": constructive_answer[:400],
                "adversarial": adversarial_answer[:400],
                "pass_number": 2,
                "phi_proxy": phi,
                "should_continue": should_continue,
                "target": target,
            })

            # Small delay to avoid rate limiting
            time.sleep(0.5)

        except Exception as e:
            print(f"  ERROR on question {i+1}: {e}")
            continue

    # Save datasets
    for filename, data in [
        ("constructive_data.jsonl", constructive_data),
        ("adversarial_data.jsonl", adversarial_data),
        ("governor_data.jsonl", governor_data),
    ]:
        path = os.path.join(output_dir, filename)
        with open(path, "w") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")
        print(f"\nSaved {len(data)} entries to {path}")

    print(f"\nGenerated {len(constructive_data)} constructive examples")
    print(f"Generated {len(adversarial_data)} adversarial examples")
    print(f"Generated {len(governor_data)} governor examples")


if __name__ == "__main__":
    if not ANTHROPIC_API_KEY:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    print(f"Generating training data for {len(TRAINING_QUESTIONS)} questions...")
    print("Using Claude as the teacher — native THEOS as the student.")
    print("Adversarial training: Socratic, collaborative, challenge that opens.\n")

    generate_data(
        questions=TRAINING_QUESTIONS,
        output_dir="./training_data",
    )
