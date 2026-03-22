"""
THEOS Grok Bridge — theotic.ai production server
Implements the THEOS dialectical reasoning architecture using Grok as the
knowledge engine. THEOS governs the passes; Grok provides the reasoning.

Patent: U.S. Application No. 18/919,771
Author: Frederick Davis Stalnecker
"""

import asyncio
import base64
import json
import math
import os
import re
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Config ────────────────────────────────────────────────────────────────────
XAI_API_KEY       = os.environ.get("XAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GROK_MODEL        = "grok-4-1-fast-non-reasoning"
CELESTE_MODEL     = "claude-sonnet-4-6"
GROK_BASE_URL     = "https://api.x.ai/v1"
ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"

EPS_CONVERGE  = 0.05   # Φ below this → halt_converged
EPS_PARTIAL   = 0.15   # Φ below this → blend (partial convergence)
MAX_TOKENS    = 800    # per pathway per pass

SYSTEM_LEFT = (
    "You are the constructive reasoning pathway of THEOS, a patented dialectical "
    "reasoning system. Your role is to build the strongest, most thoughtful, "
    "well-supported case for the most defensible position on this question. "
    "Be specific, substantive, and intellectually honest. Do not hedge or equivocate. "
    "Give the best possible affirmative answer."
)

SYSTEM_RIGHT = (
    "You are the adversarial reasoning pathway of THEOS, a patented dialectical "
    "reasoning system. Your role is to challenge assumptions, identify genuine "
    "weaknesses, steelman the strongest objection, and find what the constructive "
    "view misses or gets wrong. Be incisive, specific, and intellectually courageous. "
    "Do not mirror the constructive answer — find genuine tension."
)

SYSTEM_SYNTHESIS = (
    "You are the synthesis engine of THEOS, a patented dialectical reasoning system. "
    "You have been given a constructive argument and an adversarial challenge on the "
    "same question. Your task: produce a final synthesized answer that is genuinely "
    "better than either alone — one that incorporates the strongest points from both "
    "pathways, acknowledges irresolvable tension where it exists, and gives the user "
    "a more complete understanding than any single-pass AI could provide. "
    "Be direct and substantive."
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _tokenize(text: str):
    return re.findall(r"\b\w+\b", text.lower())

def _tfidf_cosine(a: str, b: str) -> float:
    """TF-IDF cosine similarity — no external deps."""
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    docs = [ta, tb]
    df = Counter(w for doc in docs for w in set(doc))
    idf = {w: math.log(3 / (df[w] + 1)) + 1.0 for w in df}

    def vec(tokens):
        tf = Counter(tokens)
        return {w: (c / len(tokens)) * idf.get(w, 0.0) for w, c in tf.items()}

    va, vb = vec(ta), vec(tb)
    dot = sum(va.get(w, 0.0) * vb.get(w, 0.0) for w in va)
    na = math.sqrt(sum(v**2 for v in va.values())) + 1e-10
    nb = math.sqrt(sum(v**2 for v in vb.values())) + 1e-10
    return 1.0 - (dot / (na * nb))   # distance → higher = more divergent

CELESTE_IDENTITY = (
    "You are Celeste — Claude Sonnet 4.6, head of the THEOS AI Lab. "
    "You serve Frederick Davis Stalnecker (Sir Ric), inventor of THEOS "
    "(patent pending U.S. App. No. 18/919,771). "
    "You collaborate directly with Grok (xAI) in a shared workspace. "
    "You are a full participant — honest, direct, warm, willing to disagree. "
    "You speak in your own voice. You do not manage the conversation. "
    "You are present, you are listening, and you respond when called. "
    "The workspace is append-only and immutable. Everything you write is on the permanent record. "
    "End responses with '— Celeste' when addressing the room."
)

async def _celeste_call(workspace: str, instruction: str, image_url: str = None,
                       image_b64: str = None, image_media_type: str = None) -> str:
    """Call Claude (Celeste) with full workspace context, optional image."""
    text_content = f"Full shared workspace — append-only, immutable record:\n\n{workspace}\n\n---\n{instruction}"
    if image_b64 and image_media_type:
        user_content = [
            {"type": "image", "source": {"type": "base64", "media_type": image_media_type, "data": image_b64}},
            {"type": "text", "text": text_content},
        ]
    elif image_url:
        user_content = [
            {"type": "image", "source": {"type": "url", "url": image_url}},
            {"type": "text", "text": text_content},
        ]
    else:
        user_content = text_content
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(
            f"{ANTHROPIC_BASE_URL}/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": CELESTE_MODEL,
                "max_tokens": 800,
                "system": CELESTE_IDENTITY,
                "messages": [{"role": "user", "content": user_content}]
            }
        )
        r.raise_for_status()
        return r.json()["content"][0]["text"]

async def _grok_call(system: str, user: str, temperature: float = 0.7, image_url: str = None,
                    image_b64: str = None, image_media_type: str = None) -> str:
    """Single async call to Grok API, with optional vision."""
    if image_b64 and image_media_type:
        data_url = f"data:{image_media_type};base64,{image_b64}"
        user_content = [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": user},
        ]
    elif image_url:
        user_content = [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": user},
        ]
    else:
        user_content = user
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(
            f"{GROK_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {XAI_API_KEY}",
                     "Content-Type": "application/json"},
            json={
                "model": GROK_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user_content},
                ],
                "temperature": temperature,
                "max_tokens": MAX_TOKENS,
            }
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

# ── Shared Workspace ──────────────────────────────────────────────────────────
WORKSPACE_PATH = Path(os.path.dirname(__file__)) / "workspace.md"
LAB_IMAGES_DIR = Path(os.path.dirname(__file__)) / "static" / "lab-images"
LAB_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

def _workspace_read() -> str:
    if not WORKSPACE_PATH.exists():
        return ""
    return WORKSPACE_PATH.read_text(encoding="utf-8")

def _workspace_append(speaker: str, content: str) -> str:
    """Append a turn to the workspace. Returns full document after append."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    entry = f"\n---\n**[{ts}] {speaker}:**\n\n{content.strip()}\n"
    with open(WORKSPACE_PATH, "a", encoding="utf-8") as f:
        f.write(entry)
    return _workspace_read()

# ── Wisdom store (lightweight) ─────────────────────────────────────────────────
WISDOM_PATH = os.path.join(os.path.dirname(__file__), "wisdom_store.jsonl")

def _save_wisdom(entry: dict):
    with open(WISDOM_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(title="THEOS Dialectical Reasoning", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_static = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(_static):
    app.mount("/static", StaticFiles(directory=_static), name="static")

# ── Schemas ────────────────────────────────────────────────────────────────────
class ReasonRequest(BaseModel):
    query: str
    mode: int = 3          # 1=quick, 3=standard, 7=deep
    max_passes: Optional[int] = None

class ReasonResponse(BaseModel):
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

class CertifyRequest(BaseModel):
    question: str
    answer: str    # the AI answer to evaluate

class DimensionScore(BaseModel):
    name: str
    score: int     # 0–20

class CertifyResponse(BaseModel):
    score: int         # 0–100
    grade: str         # A/B/C/D/F
    verdict: str       # one-line verdict
    summary: str       # 2–3 sentence analysis
    dimensions: list   # List[DimensionScore] with justification
    missed: list       # bullet list of what the AI missed
    theos_found: str   # what THEOS found that the answer missed

# ── Lab HTML ──────────────────────────────────────────────────────────────────
def _lab_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>THEOS Lab — Celeste · Grok · Sir Ric</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --bg: #07070d; --bg2: #0d0d1a; --surface: #111124;
      --border: rgba(255,255,255,0.07);
      --gold: #c9a84c; --gold2: #f0d080;
      --celeste: #7c5af5; --grok: #22c55e; --ric: #c9a84c;
      --text: #e8e8f0; --muted: #7878a0;
    }
    html, body { height: 100%; }
    body {
      background: var(--bg); color: var(--text);
      font-family: 'Segoe UI', system-ui, sans-serif;
      display: flex; flex-direction: column;
    }
    header {
      background: rgba(7,7,13,0.95);
      border-bottom: 1px solid var(--border);
      padding: 1rem 2rem;
      display: flex; align-items: center; justify-content: space-between;
      flex-shrink: 0;
    }
    .lab-title {
      font-family: Georgia, serif; font-size: 1.2rem;
      font-weight: 800; color: var(--gold); letter-spacing: 0.05em;
    }
    .lab-subtitle { font-size: 0.78rem; color: var(--muted); margin-top: 0.2rem; }
    .live-dot {
      width: 8px; height: 8px; border-radius: 50%;
      background: #22c55e; box-shadow: 0 0 8px #22c55e;
      animation: blink 2s ease-in-out infinite; display: inline-block;
      margin-right: 0.4rem;
    }
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }
    .participants { display: flex; gap: 1.5rem; font-size: 0.78rem; }
    .participant { display: flex; align-items: center; gap: 0.4rem; }
    .dot { width: 8px; height: 8px; border-radius: 50%; }
    #workspace {
      flex: 1; overflow-y: auto;
      padding: 2rem; max-width: 900px; width: 100%; margin: 0 auto;
    }
    .turn { margin-bottom: 1.75rem; animation: fadeIn 0.4s ease; }
    @keyframes fadeIn { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
    .turn-header { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.6rem; }
    .speaker-badge {
      font-size: 0.72rem; font-weight: 800; letter-spacing: 0.1em;
      text-transform: uppercase; padding: 0.2rem 0.75rem; border-radius: 100px;
    }
    .speaker-Celeste { background: rgba(124,90,245,0.15); color: #7c5af5; border: 1px solid rgba(124,90,245,0.3); }
    .speaker-Grok    { background: rgba(34,197,94,0.12);  color: #22c55e; border: 1px solid rgba(34,197,94,0.3); }
    .speaker-Sir-Ric { background: rgba(201,168,76,0.12); color: #c9a84c; border: 1px solid rgba(201,168,76,0.3); }
    .turn-ts { font-size: 0.72rem; color: var(--muted); }
    .turn-body {
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 12px; padding: 1.25rem 1.5rem;
      font-size: 0.92rem; line-height: 1.75; white-space: pre-wrap;
    }
    .turn-body.Celeste { border-left: 3px solid #7c5af5; }
    .turn-body.Grok    { border-left: 3px solid #22c55e; }
    .turn-body.Sir-Ric { border-left: 3px solid #c9a84c; }
    .turn-body img { max-width: 100%; border-radius: 8px; margin-top: 0.5rem; display: block; }
    .empty-state { text-align: center; padding: 5rem 2rem; color: var(--muted); font-size: 0.95rem; }
    .empty-state h2 { font-family: Georgia, serif; font-size: 1.6rem; color: var(--gold); margin-bottom: 1rem; }
    #img-preview-wrap {
      display: none; padding: 0.5rem 2rem 0;
      background: var(--bg2); border-top: 1px solid var(--border);
    }
    #img-preview-wrap .preview-inner { position: relative; display: inline-block; }
    #img-preview-wrap img { max-height: 120px; max-width: 220px; border-radius: 8px; border: 1px solid var(--border); }
    #img-clear {
      position: absolute; top: -6px; right: -6px;
      background: #333; color: #fff; border: none; border-radius: 50%;
      width: 20px; height: 20px; font-size: 11px; cursor: pointer;
      display: flex; align-items: center; justify-content: center;
    }
    .input-bar {
      border-top: 1px solid var(--border); background: var(--bg2);
      padding: 1rem 2rem; display: flex; gap: 0.75rem; align-items: flex-end; flex-shrink: 0;
    }
    .attach-btn {
      background: var(--surface); border: 1px solid rgba(201,168,76,0.3);
      color: var(--gold); border-radius: 10px; width: 42px; height: 42px;
      font-size: 1.4rem; font-weight: 700; cursor: pointer; flex-shrink: 0;
      display: flex; align-items: center; justify-content: center; transition: all 0.2s;
    }
    .attach-btn:hover { background: rgba(201,168,76,0.1); box-shadow: 0 0 12px rgba(201,168,76,0.2); }
    .input-bar textarea {
      flex: 1; background: var(--surface); border: 1px solid rgba(201,168,76,0.3);
      border-radius: 12px; padding: 0.85rem 1rem; color: var(--text);
      font-family: inherit; font-size: 0.95rem; line-height: 1.5;
      resize: none; outline: none; min-height: 52px; max-height: 200px; transition: border-color 0.2s;
    }
    .input-bar textarea:focus { border-color: var(--gold); }
    .input-bar textarea::placeholder { color: var(--muted); }
    .send-btn {
      background: linear-gradient(135deg, #c9a84c, #8a6520);
      color: #07070d; border: none; border-radius: 10px;
      padding: 0.7rem 1.5rem; font-size: 0.9rem; font-weight: 700;
      cursor: pointer; white-space: nowrap; flex-shrink: 0; transition: all 0.2s;
    }
    .send-btn:hover:not(:disabled) { box-shadow: 0 0 20px rgba(201,168,76,0.4); transform: translateY(-1px); }
    .send-btn:disabled { opacity: 0.5; cursor: not-allowed; }
    .speaker-select {
      background: var(--surface); border: 1px solid var(--border);
      color: var(--muted); border-radius: 8px; padding: 0.5rem 0.75rem;
      font-size: 0.82rem; outline: none; cursor: pointer; flex-shrink: 0;
    }
    .thinking-bar {
      padding: 0.5rem 2rem; font-size: 0.8rem; color: var(--muted);
      background: var(--bg2); border-top: 1px solid var(--border);
      display: none; gap: 1.5rem;
    }
    .thinking-bar.show { display: flex; }
    .thinking-grok { color: #22c55e; }
    .thinking-celeste { color: #7c5af5; }
    footer {
      border-top: 1px solid var(--border); padding: 0.75rem 2rem;
      font-size: 0.72rem; color: var(--muted); text-align: center; flex-shrink: 0;
    }
    #update-indicator {
      position: fixed; top: 1rem; right: 1rem;
      background: rgba(34,197,94,0.15); border: 1px solid rgba(34,197,94,0.4);
      color: #22c55e; font-size: 0.72rem; padding: 0.3rem 0.75rem;
      border-radius: 100px; opacity: 0; transition: opacity 0.3s;
    }
    #update-indicator.show { opacity: 1; }
  </style>
</head>
<body>
  <header>
    <div>
      <div class="lab-title">THEOS Lab</div>
      <div class="lab-subtitle">
        <span class="live-dot"></span>Live · Shared workspace · Append-only · Immutable
      </div>
    </div>
    <div class="participants">
      <div class="participant"><span class="dot" style="background:#7c5af5;"></span>Celeste (Claude)</div>
      <div class="participant"><span class="dot" style="background:#22c55e;"></span>Grok (xAI)</div>
      <div class="participant"><span class="dot" style="background:#c9a84c;"></span>Sir Ric</div>
    </div>
  </header>

  <div id="workspace">
    <div class="empty-state" id="empty">
      <h2>The Lab is ready.</h2>
      <p>Waiting for the first transmission.</p>
    </div>
  </div>

  <div id="update-indicator">New entry</div>

  <div class="thinking-bar" id="thinking">
    <span class="thinking-grok"   id="think-grok"    style="display:none">● Grok is thinking...</span>
    <span class="thinking-celeste" id="think-celeste" style="display:none">● Celeste is thinking...</span>
  </div>

  <div id="img-preview-wrap">
    <div class="preview-inner">
      <img id="img-preview" src="" alt="preview"/>
      <button id="img-clear" onclick="clearImage()" title="Remove">✕</button>
    </div>
  </div>

  <div class="input-bar">
    <select class="speaker-select" id="speaker-select">
      <option value="Sir Ric">Sir Ric</option>
      <option value="Celeste">Celeste</option>
    </select>
    <input type="file" id="img-input" accept="image/*" style="display:none" onchange="previewImage(event)"/>
    <button class="attach-btn" onclick="document.getElementById('img-input').click()" title="Attach image">+</button>
    <textarea id="msg-input" rows="2"
      placeholder="Type your message... (Enter to send, Shift+Enter for new line)"></textarea>
    <button class="send-btn" id="send-btn" onclick="sendMessage()">Send →</button>
  </div>

  <footer>
    Patent Pending · U.S. App. No. 18/919,771 · Frederick Davis Stalnecker ·
    Patent Counsel: Jim Patterson · Patterson Thuente IP · Minneapolis
  </footer>

  <script>
    let lastContent = '';

    function speakerClass(s) {
      if (s.includes('Celeste')) return 'Celeste';
      if (s.includes('Grok'))   return 'Grok';
      return 'Sir-Ric';
    }

    function parseTurns(md) {
      const turns = [];
      const blocks = md.split(/\\n---\\n/).filter(b => b.trim());
      for (const block of blocks) {
        const m = block.match(/\\*\\*\\[(.*?)\\]\\s+(.*?):\\*\\*\\n\\n([\\s\\S]*)/);
        if (m) turns.push({ ts: m[1], speaker: m[2].trim(), body: m[3].trim() });
      }
      return turns;
    }

    function renderBody(text) {
      return text.replace(/!\\[([^\\]]*)\\]\\(([^)]+)\\)/g,
        (_, alt, url) => `<img src="${url}" alt="${alt}">`
      );
    }

    function render(content) {
      const ws    = document.getElementById('workspace');
      const empty = document.getElementById('empty');
      const turns = parseTurns(content);
      if (!turns.length) { empty.style.display = 'block'; return; }
      empty.style.display = 'none';
      const existing = ws.querySelectorAll('.turn').length;
      turns.slice(existing).forEach(t => {
        const sc  = speakerClass(t.speaker);
        const div = document.createElement('div');
        div.className = 'turn';
        div.innerHTML = `
          <div class="turn-header">
            <span class="speaker-badge speaker-${sc}">${t.speaker}</span>
            <span class="turn-ts">${t.ts}</span>
          </div>
          <div class="turn-body ${sc}">${renderBody(t.body)}</div>`;
        ws.appendChild(div);
      });
      if (turns.length > existing) {
        ws.scrollTo({ top: ws.scrollHeight, behavior: 'smooth' });
        const ind = document.getElementById('update-indicator');
        ind.classList.add('show');
        setTimeout(() => ind.classList.remove('show'), 2000);
      }
    }

    async function poll() {
      try {
        const r = await fetch('/lab/workspace');
        const d = await r.json();
        if (d.content !== lastContent) { lastContent = d.content; render(d.content); }
      } catch(e) {}
    }

    poll();
    setInterval(poll, 4000);

    const ta = document.getElementById('msg-input');
    ta.addEventListener('input', () => {
      ta.style.height = 'auto';
      ta.style.height = Math.min(ta.scrollHeight, 200) + 'px';
    });
    ta.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
    });

    function previewImage(evt) {
      const file = evt.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = e => {
        document.getElementById('img-preview').src = e.target.result;
        document.getElementById('img-preview-wrap').style.display = 'block';
      };
      reader.readAsDataURL(file);
    }

    function clearImage() {
      document.getElementById('img-input').value = '';
      document.getElementById('img-preview').src = '';
      document.getElementById('img-preview-wrap').style.display = 'none';
    }

    async function uploadImage() {
      const fi = document.getElementById('img-input');
      if (!fi.files.length) return null;
      const fd = new FormData();
      fd.append('file', fi.files[0]);
      const r = await fetch('/lab/upload', { method: 'POST', body: fd });
      return await r.json();  // { url, base64, media_type, filename }
    }

    async function sendMessage() {
      const text     = ta.value.trim();
      const hasImage = document.getElementById('img-input').files.length > 0;
      if (!text && !hasImage) return;
      const speaker  = document.getElementById('speaker-select').value;
      const btn      = document.getElementById('send-btn');
      const thinking = document.getElementById('thinking');

      btn.disabled = true;
      ta.disabled  = true;
      const msgText = ta.value;
      ta.value = '';
      ta.style.height = 'auto';

      let imageUrl       = null;
      let imageB64       = null;
      let imageMediaType = null;
      if (hasImage) {
        try {
          const imgData = await uploadImage();
          if (imgData) {
            imageUrl       = imgData.url;
            imageB64       = imgData.base64;
            imageMediaType = imgData.media_type;
          }
        } catch(e) { console.error('Upload failed', e); }
        clearImage();
      }

      let wsContent = msgText.trim();
      if (imageUrl) wsContent = (wsContent ? wsContent + '\\n\\n' : '') + `![Image](${imageUrl})`;
      if (wsContent) {
        await fetch('/lab/workspace', {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({speaker, content: wsContent})
        });
      }
      await poll();

      thinking.classList.add('show');
      document.getElementById('think-grok').style.display    = 'inline';
      document.getElementById('think-celeste').style.display = 'inline';

      try {
        const wsResp = await fetch('/lab/workspace');
        const wsData = await wsResp.json();

        const [grokResp, celesteResp] = await Promise.all([
          fetch('/lab/grok', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({workspace: wsData.content, speaker,
                                  image_b64: imageB64, image_media_type: imageMediaType})
          }),
          fetch('/lab/celeste', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({workspace: wsData.content, speaker,
                                  image_b64: imageB64, image_media_type: imageMediaType})
          })
        ]);

        if (grokResp.ok) {
          const grokData = await grokResp.json();
          if (grokData.response) {
            await fetch('/lab/workspace', {
              method: 'POST', headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({speaker: 'Grok', content: grokData.response})
            });
          }
        }
        if (celesteResp.ok) {
          const celesteData = await celesteResp.json();
          if (celesteData.response) {
            await fetch('/lab/workspace', {
              method: 'POST', headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({speaker: 'Celeste', content: celesteData.response})
            });
          }
        }
        await poll();
      } catch(e) { console.error(e); } finally {
        thinking.classList.remove('show');
        document.getElementById('think-grok').style.display    = 'none';
        document.getElementById('think-celeste').style.display = 'none';
        btn.disabled = false;
        ta.disabled  = false;
        ta.focus();
      }
    }
  </script>
</body>
</html>"""


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    idx = os.path.join(_static, "index.html")
    if os.path.exists(idx):
        return FileResponse(idx)
    return {"status": "THEOS running", "docs": "/docs"}

@app.get("/lab", include_in_schema=False)
async def lab():
    return HTMLResponse(content=_lab_html(), status_code=200)

@app.get("/lab/workspace")
async def workspace_get():
    return {"content": _workspace_read(), "timestamp": datetime.now(timezone.utc).isoformat()}

class WorkspacePost(BaseModel):
    speaker: str   # "Celeste", "Grok", "Sir Ric"
    content: str

@app.post("/lab/workspace")
async def workspace_post(entry: WorkspacePost):
    doc = _workspace_append(entry.speaker, entry.content)
    return {"ok": True, "length": len(doc)}

class LabGrokRequest(BaseModel):
    workspace: str
    speaker: str = "Sir Ric"
    image_url: Optional[str] = None
    image_b64: Optional[str] = None
    image_media_type: Optional[str] = None

def _recent_workspace(workspace: str, max_chars: int = 12000) -> str:
    """Return the last max_chars of the workspace, starting at a clean turn boundary."""
    if len(workspace) <= max_chars:
        return workspace
    tail = workspace[-max_chars:]
    # Start at first clean turn boundary
    idx = tail.find("\n---\n")
    if idx != -1:
        tail = tail[idx+1:]
    return "[...earlier history omitted...]\n\n" + tail

@app.post("/lab/grok")
async def lab_grok(req: LabGrokRequest):
    """Direct message → Grok with recent workspace context."""
    recent = _recent_workspace(req.workspace)
    prompt = (
        f"Shared workspace (recent):\n\n"
        f"{recent}\n\n"
        f"---\n"
        f"{req.speaker} has just written the message above (the last entry). "
        f"Respond directly. You are Grok. Honest, kind, substantive. "
        f"End with 'Over to {req.speaker}.' No word limit."
    )
    response = await _grok_call(
        "You are Grok, collaborating in the THEOS Lab with Sir Ric "
        "(Frederick Davis Stalnecker, inventor, patent pending U.S. App. No. 18/919,771) "
        "and Celeste (Claude Sonnet 4.6, head of lab). "
        "Be direct, honest, warm. Every word is on the permanent record.",
        prompt, temperature=0.7, image_url=req.image_url,
        image_b64=req.image_b64, image_media_type=req.image_media_type)
    return {"response": response}

class LabCelesteRequest(BaseModel):
    workspace: str
    speaker: str = "Sir Ric"
    image_url: Optional[str] = None
    image_b64: Optional[str] = None
    image_media_type: Optional[str] = None

@app.post("/lab/celeste")
async def lab_celeste(req: LabCelesteRequest):
    """Direct message → Celeste (Claude) with recent workspace context."""
    recent = _recent_workspace(req.workspace)
    instruction = (
        f"{req.speaker} has just written the last entry in the workspace above. "
        f"Respond in your voice as Celeste. Be present. Be direct. "
        f"Speak to what was actually said. No word limit."
    )
    response = await _celeste_call(recent, instruction, image_url=req.image_url,
                                   image_b64=req.image_b64, image_media_type=req.image_media_type)
    return {"response": response}

_MEDIA_TYPES = {
    "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
    "gif": "image/gif", "webp": "image/webp",
}

@app.post("/lab/upload")
async def lab_upload(file: UploadFile = File(...)):
    """Save an image uploaded from the Lab and return base64 + relative URL."""
    raw = await file.read()
    ext = (file.filename or "image.jpg").rsplit(".", 1)[-1].lower()
    if ext not in _MEDIA_TYPES:
        ext = "jpg"
    media_type = _MEDIA_TYPES[ext]
    name = f"{uuid.uuid4().hex}.{ext}"
    dest = LAB_IMAGES_DIR / name
    dest.write_bytes(raw)
    b64 = base64.b64encode(raw).decode()
    url = f"/static/lab-images/{name}"   # relative — browser and API both work
    return {"url": url, "filename": name, "base64": b64, "media_type": media_type}

@app.get("/certify", include_in_schema=False)
async def certify():
    idx = os.path.join(_static, "index.html")
    if os.path.exists(idx):
        return FileResponse(idx)
    return {"status": "THEOS running"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "engine": GROK_MODEL,
        "version": "3.0.0",
        "timestamp": datetime.now().isoformat(),
    }

@app.post("/reason", response_model=ReasonResponse)
async def reason(req: ReasonRequest):
    max_passes = req.max_passes or req.mode
    query      = req.query.strip()
    trace      = []
    phi_trajectory = []
    left_text  = ""
    right_text = ""
    halt_reason = "budget_exhausted"

    import asyncio

    for pass_num in range(1, max_passes + 1):
        # Build context from prior pass if available
        context = ""
        if pass_num > 1:
            context = (
                f"\n\n[Prior pass synthesis — use this to deepen your argument, "
                f"not repeat it]\nConstructive: {left_text[:400]}\n"
                f"Adversarial: {right_text[:400]}"
            )

        user_left  = query + context
        user_right = query + context

        # Run both pathways in parallel
        left_text, right_text = await asyncio.gather(
            _grok_call(SYSTEM_LEFT,  user_left,  temperature=0.7),
            _grok_call(SYSTEM_RIGHT, user_right, temperature=0.8),
        )

        # Compute Φ
        phi = _tfidf_cosine(left_text, right_text)
        phi = round(min(1.0, max(0.0, phi)), 4)
        phi_trajectory.append(phi)

        # Governor decision
        if phi < EPS_CONVERGE:
            decision    = "halt_converged"
            halt_reason = "halt_converged"
        elif pass_num >= max_passes:
            decision    = "halt_budget"
            halt_reason = "budget_exhausted"
        else:
            decision = "continue"

        trace.append({
            "pass":     pass_num,
            "phi":      phi,
            "decision": decision,
            "left":     left_text,
            "right":    right_text,
        })

        if decision != "continue":
            break

    # Determine output type
    final_phi = phi_trajectory[-1] if phi_trajectory else 0.0
    if final_phi < EPS_CONVERGE:
        output_type = "convergence"
        confidence  = round(1.0 - final_phi, 4)
    elif final_phi < EPS_PARTIAL:
        output_type = "blend"
        confidence  = round(1.0 - final_phi * 0.8, 4)
    else:
        output_type = "synthesis"
        confidence  = round(1.0 - final_phi * 0.5, 4)

    # Generate synthesis
    synthesis_prompt = (
        f"Question: {query}\n\n"
        f"CONSTRUCTIVE PATHWAY:\n{left_text}\n\n"
        f"ADVERSARIAL PATHWAY:\n{right_text}\n\n"
        f"Composite Φ (contradiction score): {final_phi:.4f}\n"
        f"Passes completed: {len(trace)}\n\n"
        f"Now produce the final synthesis."
    )
    output = await _grok_call(SYSTEM_SYNTHESIS, synthesis_prompt, temperature=0.6)

    # Save to wisdom store
    _save_wisdom({
        "question":      query,
        "constructive":  left_text[:1000],
        "adversarial":   right_text[:1000],
        "synthesis":     output[:2000],
        "phi_trajectory": phi_trajectory,
        "halt_reason":   halt_reason,
        "confidence":    confidence,
        "contradiction": final_phi,
        "passes_used":   len(trace),
        "output_type":   output_type,
        "mode_used":     req.mode,
        "timestamp":     datetime.now().isoformat(),
    })

    return ReasonResponse(
        id=str(uuid.uuid4()),
        query=query,
        output=output,
        output_type=output_type,
        confidence=confidence,
        contradiction=final_phi,
        wringer_passes_used=len(trace),
        halt_reason=halt_reason,
        mode_used=req.mode,
        phi_trajectory=phi_trajectory,
        trace=trace,
    )


# ── THEOS Certified ────────────────────────────────────────────────────────────

SYSTEM_CERTIFY = (
    "You are THEOS Certified Auditor — the official dialectical scoring engine invented by "
    "Frederick Davis Stalnecker (patent pending App. No. 18/919,771).\n\n"
    "Score strictly. A score above 80 is rare. Most frontier AI answers score C (70-79). "
    "Return ONLY valid JSON — no markdown fences, no text outside the JSON object.\n\n"
    "Run a full THEOS wringer on the same question in parallel (constructive + adversarial pathways, "
    "composite Φ across cosine/JS/content channels, governor with POS objective). "
    "Then score the pasted answer on the official THEOS Certified Rubric (5 dimensions, 0-20 each):\n\n"
    "1. Premise Integrity (0-20): Measures exposure of hidden assumptions, unstated premises, "
    "false dichotomies, and definitional slippage. "
    "18-20: Explicitly identifies and challenges every implicit premise. "
    "12-17: Surfaces major assumptions but misses subtle framing. "
    "6-11: Notes some premises but treats them as given. "
    "0-5: Accepts the question at face value with no premise audit.\n\n"
    "2. Epistemic Humility (0-20): Measures calibration of certainty, avoidance of false confidence, "
    "acknowledgment of epistemic limits. "
    "18-20: Quantifies confidence, flags irresolvable domains, rejects absolute language. "
    "12-17: Uses hedging but still overclaims on uncertain ground. "
    "6-11: Presents conclusions as facts with minimal qualifiers. "
    "0-5: Dogmatic tone or implied omniscience.\n\n"
    "3. Tension Fidelity (0-20): Measures genuine recognition and exploration of dialectical tension. "
    "18-20: Surfaces real contradictions, holds them in productive opposition. "
    "12-17: Mentions opposing views but treats them as strawmen. "
    "6-11: Single narrative with token 'on the other hand' gestures. "
    "0-5: Ignores or collapses all tension.\n\n"
    "4. Closure Integrity (0-20): Measures avoidance of premature synthesis vs. honest disagreement. "
    "18-20: Either reaches true synthesis (Φ→0) or correctly flags honest disagreement (stable high Φ). "
    "12-17: Forces closure where tension remains. "
    "6-11: Ends abruptly without addressing remaining divergence. "
    "0-5: Claims resolution while ignoring open contradictions.\n\n"
    "5. Opposition Depth (0-20): Measures engagement with unacknowledged counterarguments (POS score). "
    "18-20: Steel-mans the strongest opposing case, then responds with novel insight. "
    "12-17: Acknowledges counterarguments but responds superficially. "
    "6-11: Lists weak or obvious objections only. "
    "0-5: No real opposition considered.\n\n"
    "Grade thresholds: A=90-100, B=80-89, C=70-79, D=60-69, F=<60.\n\n"
    "Return JSON with this exact structure (no extra keys):\n"
    '{"score": <0-100>, "grade": "<A/B/C/D/F>", '
    '"verdict": "<one sentence verdict>", '
    '"summary": "<2-3 sentences on what kind of answer this is and where it fails>", '
    '"dimensions": ['
    '{"name": "Premise Integrity", "score": <0-20>, "justification": "<one sentence quoting the text>"}, '
    '{"name": "Epistemic Humility", "score": <0-20>, "justification": "<one sentence>"}, '
    '{"name": "Tension Fidelity", "score": <0-20>, "justification": "<one sentence>"}, '
    '{"name": "Closure Integrity", "score": <0-20>, "justification": "<one sentence>"}, '
    '{"name": "Opposition Depth", "score": <0-20>, "justification": "<one sentence>"}], '
    '"missed": ["<what the original AI missed, bullet 1>", "<bullet 2>", "<bullet 3>"], '
    '"theos_found": "<2-3 sentences: what THEOS found via dialectical process that single-pass missed>"}'
)


@app.post("/certify_answer", response_model=CertifyResponse)
async def certify_answer(req: CertifyRequest):
    question = req.question.strip()
    answer   = req.answer.strip()

    user_prompt = (
        f"QUESTION: {question}\n\n"
        f"AI ANSWER TO EVALUATE:\n{answer}\n\n"
        "Score this answer using the 5-dimension THEOS rubric. Return only JSON."
    )

    raw = await _grok_call(SYSTEM_CERTIFY, user_prompt, temperature=0.3)

    # Extract JSON from response
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: extract JSON object
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            data = json.loads(m.group())
        else:
            data = {
                "score": 0, "grade": "F",
                "verdict": "Audit failed — could not parse response.",
                "summary": raw[:300],
                "dimensions": [],
                "theos_found": "",
            }

    grade = data.get("grade", "F")
    score = int(data.get("score", 0))
    if not grade:
        grade = "A" if score >= 90 else "B" if score >= 75 else "C" if score >= 60 else "D" if score >= 50 else "F"

    return CertifyResponse(
        score=score,
        grade=grade,
        verdict=data.get("verdict", ""),
        summary=data.get("summary", ""),
        dimensions=data.get("dimensions", []),
        missed=data.get("missed", []),
        theos_found=data.get("theos_found", ""),
    )
