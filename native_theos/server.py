"""
THEOS Native API Server
========================

FastAPI server exposing native THEOS reasoning as a REST API.

Endpoints:
    POST /reason              — Submit a query for THEOS reasoning
    GET  /health              — Health check with model status
    GET  /wisdom              — Wisdom store statistics
    GET  /trace/{run_id}      — Full trace for a reasoning run
    POST /audit               — Audit an answer from another AI system

Reasoning Commons (new):
    POST /commons/submit      — Submit wisdom to the shared Commons
    GET  /commons/search      — Search the Commons by query
    GET  /commons/stats       — Commons statistics
    POST /certify             — Full THEOS certification with structured scoring

The Reasoning Commons is a quality-gated shared wisdom layer across all
THEOS-certified reasoning instances. Each entry requires genuine dialectical
tension (phi > 0.01) and confident resolution (confidence > 0.60). Any AI
can search the Commons before answering; any THEOS instance can contribute.

The /certify endpoint is the THEOS Certified capability:
    Submit any AI-generated answer. THEOS runs its full wringer on
    the same question and returns: a structured certification score (0-100),
    grade (A-F), specific gaps found, Phi score, and synthesis.

Patent: U.S. Application No. 18/919,771 (Provisional #63/831,738)
Author: Frederick Davis Stalnecker
"""

import asyncio
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Lazy imports for model-dependent modules (torch not available on CPU-only servers)
try:
    from architecture import NativeTheos, NativeTheosConfig
    from inference import run_native_wringer
    _MODEL_AVAILABLE = True
    # Allow BASE_MODEL_PATH env var to override HuggingFace download (needed on DO droplet)
    _base_model = os.environ.get("BASE_MODEL_PATH", "meta-llama/Meta-Llama-3-8B-Instruct")
    _model_config = NativeTheosConfig(base_model_name=_base_model)
except ImportError as _e:
    print(f"Model dependencies not available ({_e}) — model endpoints will return 503.")
    NativeTheos = None
    NativeTheosConfig = None
    run_native_wringer = None
    _MODEL_AVAILABLE = False
    _model_config = None

from wisdom import WisdomStore
from reasoning_commons import ReasoningCommons, compute_certification_score


def _build_wisdom_context(ws: "WisdomStore", query: str) -> str:
    """Build a wisdom context string from past entries for injection into prompts."""
    try:
        past = ws.retrieve(query, max_results=2)
    except Exception:
        return ""
    if not past:
        return ""
    snippets = []
    for p in past:
        q_text   = p.get("question", "")
        synth    = p.get("synthesis", p.get("output", ""))[:400]
        hr       = p.get("halt_reason", "")
        conf     = p.get("confidence", 0.0)
        snippets.append(
            f"Past query: {q_text}\n"
            f"Synthesis ({hr}, confidence={conf:.0%}):\n{synth}"
        )
    return "\n\n---\n\n".join(snippets)

# Single-worker executor: only one inference at a time (CPU memory constraint)
# Keeps event loop unblocked so /health still responds during long inference.
_inference_executor = ThreadPoolExecutor(max_workers=1)
_inference_busy = False

# Data directory — can be overridden via THEOS_DATA_DIR env var
_data_dir = os.environ.get("THEOS_DATA_DIR", os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# App initialization
# ============================================================

app = FastAPI(
    title="THEOS Native Reasoning API",
    description=(
        "Dual-engine dialectical reasoning with governed contradiction. "
        "Patent U.S. 18/919,771. Includes the Reasoning Commons — "
        "a shared quality-gated wisdom layer for THEOS-certified AI."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded model, wisdom store, and Reasoning Commons
model = None
config = _model_config
wisdom_store = WisdomStore(store_path=os.path.join(_data_dir, "wisdom_store.jsonl"))
commons = ReasoningCommons(store_path=os.path.join(_data_dir, "commons_store.jsonl"))
reasoning_history = {}


# ============================================================
# Request / Response models
# ============================================================

class ReasoningRequest(BaseModel):
    query: str
    mode: Optional[int] = None          # 1=Reflective, 2=Dialectical, 3=Full Wringer
    max_passes: Optional[int] = None


class AuditRequest(BaseModel):
    question: str
    answer_to_audit: str
    source: Optional[str] = "external_ai"


class ReasoningResponse(BaseModel):
    id: str
    query: str
    output: str
    output_type: str
    confidence: float
    contradiction: float
    wringer_passes_used: int
    halt_reason: str
    mode_used: int
    phi_trajectory: list
    trace: list
    timestamp: str


class AuditResponse(BaseModel):
    id: str
    question: str
    original_answer: str
    theos_output: str
    theos_output_type: str
    phi_score: float
    what_was_missed: str
    confidence: float
    certified: bool                     # True if THEOS agrees with the original answer
    timestamp: str


class CertifyRequest(BaseModel):
    question: str
    answer_to_certify: str
    source: Optional[str] = "external_ai"


class CertifyResponse(BaseModel):
    id: str
    question: str
    original_answer: str
    certification_score: int            # 0-100
    grade: str                          # A/B/C/D/F
    verdict: str                        # One-sentence summary
    gaps: List[str]                     # Specific gaps found
    subscores: dict                     # Dialectical / resolution / epistemic
    theos_synthesis: str
    theos_phi_score: float
    theos_confidence: float
    theos_halt_reason: str
    source: str
    timestamp: str


class CommonsSubmitRequest(BaseModel):
    question: str
    constructive: str
    adversarial: str
    synthesis: str
    phi_trajectory: List[float]
    halt_reason: str
    confidence: float
    phi_composite: float
    passes_used: int
    source: Optional[str] = "external_theos"
    domain: Optional[str] = "general"


class CommonsSearchResponse(BaseModel):
    query: str
    results: List[dict]
    total_commons_entries: int


# ============================================================
# Startup
# ============================================================

@app.on_event("startup")
async def startup():
    global model
    if not _MODEL_AVAILABLE:
        print("PyTorch not available — model endpoints disabled. Commons endpoints active.")
        return
    print("Loading THEOS Native model...")
    try:
        model = NativeTheos(config)
        print("THEOS Native ready.")
    except Exception as e:
        print(f"Model load failed: {e}")
        print("Server running without model — /reason will return 503 until model loads.")
        model = None


# ============================================================
# Endpoints
# ============================================================

@app.post("/reason", response_model=ReasoningResponse)
async def reason(request: ReasoningRequest):
    """Submit a query for THEOS native reasoning."""
    global _inference_busy
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if _inference_busy:
        raise HTTPException(status_code=429, detail="Inference in progress — try again shortly")

    # Note: request.max_passes is not yet wired into run_native_wringer (uses model.config.mode{n}_max_passes)
    wctx = _build_wisdom_context(wisdom_store, request.query)
    _inference_busy = True
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _inference_executor,
            lambda: run_native_wringer(
                model, request.query, mode=request.mode, verbose=False,
                wisdom_context=wctx or None,
            ),
        )
    finally:
        _inference_busy = False

    # Accumulate wisdom
    left_text = result.trace[0].left_output if result.trace else ""
    right_text = result.trace[0].right_output if result.trace else ""

    wisdom_store.add(
        question=request.query,
        constructive=left_text,
        adversarial=right_text,
        synthesis=result.output,
        phi_trajectory=result.phi_trajectory_composite,
        halt_reason=result.halt_reason,
        confidence=result.confidence,
        contradiction=result.contradiction,
        passes_used=result.wringer_passes_used,
        mode_used=result.mode_used,
    )

    # Auto-submit to Reasoning Commons (quality-gated — low-quality runs rejected)
    commons.submit(
        question=request.query,
        constructive=left_text,
        adversarial=right_text,
        synthesis=result.output,
        phi_trajectory=result.phi_trajectory_composite,
        halt_reason=result.halt_reason,
        confidence=result.confidence,
        phi_composite=result.contradiction,
        passes_used=result.wringer_passes_used,
        source="theos_native",
        mode_used=result.mode_used,
    )

    run_id = str(uuid.uuid4())
    response = ReasoningResponse(
        id=run_id,
        query=request.query,
        output=result.output,
        output_type=result.output_type,
        confidence=result.confidence,
        contradiction=result.contradiction,
        wringer_passes_used=result.wringer_passes_used,
        halt_reason=result.halt_reason,
        mode_used=result.mode_used,
        phi_trajectory=result.phi_trajectory_composite,
        trace=[
            {
                "pass": t.wringer_pass + 1,
                "phi": t.phi_composite,
                "decision": t.governor_decision,
                "left": t.left_output[:200],
                "right": t.right_output[:200],
            }
            for t in result.trace
        ],
        timestamp=datetime.now().isoformat(),
    )

    reasoning_history[run_id] = response
    return response


@app.post("/audit", response_model=AuditResponse)
async def audit(request: AuditRequest):
    """
    THEOS Certified: audit an answer from another AI system.

    THEOS runs its full wringer on the same question and reports:
    - Phi score (how much genuine opposition exists around this answer)
    - What the adversarial engine found that the original answer missed
    - Whether THEOS agrees with the original answer
    """
    global _inference_busy
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if _inference_busy:
        raise HTTPException(status_code=429, detail="Inference in progress — try again shortly")

    _inference_busy = True
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _inference_executor,
            lambda: run_native_wringer(model, request.question, mode=3, verbose=False),
        )
    finally:
        _inference_busy = False

    # The adversarial output is what was missed
    what_was_missed = ""
    if result.trace:
        what_was_missed = result.trace[0].right_output

    # Certified = THEOS converged and agrees with the original answer's direction
    certified = (
        result.output_type == "convergence" and
        result.confidence > 0.7
    )

    run_id = str(uuid.uuid4())
    response = AuditResponse(
        id=run_id,
        question=request.question,
        original_answer=request.answer_to_audit,
        theos_output=result.output,
        theos_output_type=result.output_type,
        phi_score=result.contradiction,
        what_was_missed=what_was_missed,
        confidence=result.confidence,
        certified=certified,
        timestamp=datetime.now().isoformat(),
    )

    reasoning_history[run_id] = response
    return response


@app.post("/certify", response_model=CertifyResponse)
async def certify(request: CertifyRequest):
    """
    THEOS Certified: structured certification for any AI's answer.

    Runs the full wringer on the question, then scores the submitted answer
    against THEOS's reasoning on three dimensions:
    - Dialectical completeness (40 pts): did it address opposing framings?
    - Resolution quality (30 pts): does synthesis align with THEOS?
    - Epistemic honesty (30 pts): correct uncertainty where needed?

    Returns a score (0-100), grade (A-F), and specific gaps identified.
    """
    global _inference_busy
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if _inference_busy:
        raise HTTPException(status_code=429, detail="Inference in progress — try again shortly")

    _inference_busy = True
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _inference_executor,
            lambda: run_native_wringer(model, request.question, mode=3, verbose=False),
        )
    finally:
        _inference_busy = False

    left_text = result.trace[0].left_output if result.trace else ""
    right_text = result.trace[0].right_output if result.trace else ""

    # Compute certification score
    cert = compute_certification_score(
        theos_phi=result.contradiction,
        theos_confidence=result.confidence,
        theos_halt_reason=result.halt_reason,
        external_answer=request.answer_to_certify,
        theos_synthesis=result.output,
        theos_adversarial=right_text,
    )

    # Store in commons with certification metadata
    commons.submit(
        question=request.question,
        constructive=left_text,
        adversarial=right_text,
        synthesis=result.output,
        phi_trajectory=result.phi_trajectory_composite,
        halt_reason=result.halt_reason,
        confidence=result.confidence,
        phi_composite=result.contradiction,
        passes_used=result.wringer_passes_used,
        source=request.source,
        external_answer=request.answer_to_certify,
        certification_result=cert,
    )

    run_id = str(uuid.uuid4())
    response = CertifyResponse(
        id=run_id,
        question=request.question,
        original_answer=request.answer_to_certify,
        certification_score=cert["certification_score"],
        grade=cert["grade"],
        verdict=cert["verdict"],
        gaps=cert["gaps"],
        subscores=cert["subscores"],
        theos_synthesis=result.output,
        theos_phi_score=result.contradiction,
        theos_confidence=result.confidence,
        theos_halt_reason=result.halt_reason,
        source=request.source,
        timestamp=datetime.now().isoformat(),
    )

    reasoning_history[run_id] = response
    return response


@app.post("/commons/submit")
async def commons_submit(request: CommonsSubmitRequest):
    """
    Submit reasoning to the Reasoning Commons from any THEOS instance.

    Quality-gated: entries require confidence > 0.60 and phi > 0.01.
    This endpoint enables federated Commons — multiple THEOS instances
    contributing to a shared dialectical wisdom pool.
    """
    entry_id = commons.submit(
        question=request.question,
        constructive=request.constructive,
        adversarial=request.adversarial,
        synthesis=request.synthesis,
        phi_trajectory=request.phi_trajectory,
        halt_reason=request.halt_reason,
        confidence=request.confidence,
        phi_composite=request.phi_composite,
        passes_used=request.passes_used,
        source=request.source,
        domain=request.domain,
    )
    if not entry_id:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Entry rejected: confidence ({request.confidence:.2f}) or "
                f"phi ({request.phi_composite:.4f}) below quality threshold. "
                f"Commons requires confidence >= {ReasoningCommons.MIN_CONFIDENCE} "
                f"and phi >= {ReasoningCommons.MIN_PHI}."
            ),
        )
    return {"id": entry_id, "status": "admitted", "total_entries": len(commons.entries)}


@app.get("/commons/search", response_model=CommonsSearchResponse)
async def commons_search(
    q: str = Query(..., description="Search query"),
    max_results: int = Query(5, ge=1, le=20),
    domain: Optional[str] = Query(None),
    min_phi: float = Query(0.0, ge=0.0),
):
    """
    Search the Reasoning Commons for relevant past THEOS reasoning.

    Any AI system can call this before answering a question to retrieve
    what THEOS previously found when the wringer was run on similar questions.
    This seeds richer reasoning without requiring a full wringer pass.
    """
    results = commons.search(q, max_results=max_results, domain=domain, min_phi=min_phi)
    return CommonsSearchResponse(
        query=q,
        results=[
            {
                "question": e["question"],
                "synthesis": e["synthesis"][:500],
                "adversarial_framing": e["adversarial"][:300],
                "phi_composite": e.get("phi_composite", 0),
                "confidence": e.get("confidence", 0),
                "halt_reason": e.get("halt_reason", ""),
                "passes": e.get("passes_used", 0),
                "source": e.get("source", ""),
                "timestamp": e.get("timestamp", ""),
            }
            for e in results
        ],
        total_commons_entries=len(commons.entries),
    )


@app.get("/commons/stats")
async def commons_stats():
    """Reasoning Commons statistics — entries, quality metrics, source distribution."""
    return commons.get_stats()


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_available": _MODEL_AVAILABLE,
        "inference_busy": _inference_busy,
        "model_name": config.base_model_name if (model and config) else None,
        "wisdom_entries": len(wisdom_store.entries),
        "commons_entries": len(commons.entries),
        "version": "2.1.0",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/wisdom")
async def get_wisdom():
    return wisdom_store.get_stats()


@app.get("/trace/{run_id}")
async def get_trace(run_id: str):
    if run_id not in reasoning_history:
        raise HTTPException(status_code=404, detail="Run not found")
    return reasoning_history[run_id]


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
