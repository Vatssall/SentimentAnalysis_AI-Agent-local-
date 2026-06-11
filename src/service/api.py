# src/service/api.py
from __future__ import annotations
from typing import Dict, Optional, List, Any
from datetime import datetime
import numbers
import traceback

from fastapi import FastAPI, HTTPException, Depends, Request, Path
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy.orm import Session
from ollama import chat

from ..config import ROOT
from .models_registry import get as get_model, available as available_tasks, load_models
from .db import init_db, SessionLocal, User, ChatSession, Message, Memory
from .auth import get_db, hash_password, verify_password, create_access_token, get_current_user

# therapist helper (only for generating suggestions, no separate tab/endpoints)
from ..therapy.plans import suggest_tasks
from ..therapy.safety import detect_risk

app = FastAPI(title="Local Agent • Unified Chat w/ Therapy Suggestions")

# --- static + templates ---
STATIC_DIR = (ROOT / "static"); STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR = (ROOT / "templates"); TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# --- db init ---
init_db()

@app.on_event("startup")
def _startup_load_models():
    tasks = load_models()
    print(f"[STARTUP] Loaded tasks: {tasks}")

# ---------- Schemas ----------
class PredictIn(BaseModel): text: str
class PredictOut(BaseModel): task: str; label: str; proba: float
class AuthIn(BaseModel): email: str; password: str
class ChatIn(BaseModel): text: str; session_id: Optional[int] = None
class ChatOut(BaseModel):
    session_id: int
    reply: str
    signals: Dict
    tasks: Optional[List[Dict]] = None
    risky: bool = False

# ---------- Utils ----------
def _to_py(v: Any) -> Any:
    """Convert numpy/torch types into plain Python so FastAPI can JSON-serialize."""
    try:
        import numpy as np
        if isinstance(v, (np.floating, np.integer)): return v.item()
        if isinstance(v, np.ndarray): return [_to_py(x) for x in v.tolist()]
    except Exception:
        pass
    try:
        import torch
        if isinstance(v, torch.Tensor):
            if v.ndim == 0: return v.item()
            return [_to_py(x) for x in v.detach().cpu().numpy().tolist()]
    except Exception:
        pass
    if isinstance(v, (list, tuple)): return [_to_py(x) for x in v]
    if isinstance(v, dict): return {str(k): _to_py(val) for k, val in v.items()}
    if isinstance(v, numbers.Number): return float(v) if isinstance(v, numbers.Real) else v
    return v

def classify_with_local_models(text: str) -> Dict:
    """Run available models; return only tasks that produced a label."""
    results: Dict[str, dict] = {}
    for task in ("sentiment", "emotion", "mood"):
        bundle = get_model(task)
        if not bundle: 
            continue
        try:
            X = [text]
            probs_dict = {}
            label = None
            proba = 0.0

            if "hf" in bundle:
                hf = bundle["hf"]
                labels, probs = hf.predict(X)            # probs may be numpy/torch
                probs = _to_py(probs)
                # safety: zip so we never index out of range
                for l, p in zip(hf.labels, probs[0]):
                    probs_dict[str(l)] = float(p)
                if probs_dict:
                    label = max(probs_dict, key=probs_dict.get)
                    proba = float(probs_dict[label])

            elif "skl" in bundle:
                pipe = bundle["skl"]["pipeline"]
                label = str(pipe.predict(X)[0])
                if hasattr(pipe, "predict_proba"):
                    p = _to_py(pipe.predict_proba(X)[0])
                    classes = list(getattr(pipe[-1], "classes_", []))
                    for c, pv in zip(classes, p):
                        probs_dict[str(c)] = float(pv)
                    if probs_dict:
                        label = max(probs_dict, key=probs_dict.get)
                        proba = float(probs_dict[label])

            if label:
                results[task] = {
                    "task": task,
                    "label": str(label),
                    "proba": float(proba),
                    "probs": probs_dict,
                }
        except Exception as e:
            print(f"[PREDICT][{task}] error: {e}")
            traceback.print_exc()
            continue
    return results

NEG_SENTIMENTS = {"anxiety", "stress", "depression", "suicidal", "negative", "distress"}
NEG_EMOTIONS   = {"sad", "fear", "anger", "disgust", "shame", "guilt", "worry", "confusion"}

def is_negative(signals: Dict, text: str) -> bool:
    s = (signals.get("sentiment") or {})
    e = (signals.get("emotion") or {})
    m = (signals.get("mood") or {})

    neg = s.get("label","") in {"anxiety","stress","depression","suicidal","negative"} \
          or e.get("label","") in {"fear","worry","anger","sadness","shame","guilt","disgust"} \
          or m.get("label","") in {"nervous","tense","sad","angry","discouraged"}

    # require some confidence if the label came from Llama
    conf = max(float(s.get("proba",0.0)), float(e.get("proba",0.0)), float(m.get("proba",0.0)))
    return neg and conf >= 0.6 or detect_risk(text, signals)

def build_memory_summary(db: Session, user: User, k: int = 20) -> str:
    last_msgs = (
        db.query(Message)
        .join(ChatSession, Message.session_id == ChatSession.id)
        .filter(ChatSession.user_id == user.id)
        .order_by(Message.created_at.desc())
        .limit(k)
        .all()
    )
    last_msgs = list(reversed(last_msgs))
    lines = []
    for m in last_msgs:
        s = m.signals or {}
        sig = ", ".join(f"{t}:{s.get(t,{}).get('label','')}" for t in ("sentiment","emotion","mood") if t in s)
        lines.append(f"{m.role.upper()}: {m.content}  [{sig}]")
    mems = db.query(Memory).filter(Memory.user_id == user.id).all()
    mlines = [f"{m.key}={m.value}" for m in mems]
    return "Recent conversation:\n" + "\n".join(lines) + ("\n\nUser memory: " + "; ".join(mlines) if mlines else "")

def update_rolling_memory(db: Session, user: User, signals: Dict):
    for t in ("sentiment","emotion","mood"):
        label = signals.get(t,{}).get("label")
        if not label: continue
        mem = db.query(Memory).filter(Memory.user_id==user.id, Memory.key==f"last_{t}").first()
        if not mem: mem = Memory(user_id=user.id, key=f"last_{t}", value={"label": label}); db.add(mem)
        else: mem.value = {"label": label}
    db.commit()

def _session_summary(s: ChatSession) -> dict:
    last = s.messages[-1].created_at.isoformat() if s.messages else s.created_at.isoformat()
    preview = (s.messages[-1].content[:60] + "…") if s.messages else ""
    return {"id": s.id, "title": s.title, "created_at": s.created_at.isoformat(),
            "last_message_at": last, "preview": preview}

import time
from datetime import timedelta

# --- signal gating / cooldowns ---
SHOW_CHIPS_ON_FIRST = True
PROBA_DELTA_TO_SHOW = 0.20     # how much the top proba must change to show chips again
TASK_COOLDOWN_MIN   = 8        # minimum minutes between task suggestions per user

def _top(sig: Dict) -> tuple[str, float]:
    """Return (label, proba) from one task signal dict {'label':..., 'proba':..., 'probs':{...}}."""
    if not sig: return ("", 0.0)
    lbl = str(sig.get("label","") or "")
    p   = float(sig.get("proba") or 0.0)
    return (lbl, p)

def _signals_summary(signals: Dict) -> Dict:
    """Compact summary: only the winning label + proba per task, so we can compare easily."""
    out={}
    for t in ("sentiment","emotion","mood"):
        if t in signals:
            lbl, p = _top(signals[t])
            out[t] = {"label": lbl, "proba": p}
    return out

def _signals_changed(old: Dict, new: Dict) -> bool:
    """Return True if any label changed OR top proba changed by >= threshold."""
    for t in ("sentiment","emotion","mood"):
        o = old.get(t) or {}
        n = new.get(t) or {}
        if not o and n: return True
        if not n and o: return False
        if (o.get("label") or "") != (n.get("label") or ""):
            return True
        if abs(float(o.get("proba") or 0.0) - float(n.get("proba") or 0.0)) >= PROBA_DELTA_TO_SHOW:
            return True
    return False

def _mem_get(db: Session, user: User, key: str, default=None):
    m = db.query(Memory).filter(Memory.user_id==user.id, Memory.key==key).first()
    return m.value if m else default

def _mem_set(db: Session, user: User, key: str, value):
    m = db.query(Memory).filter(Memory.user_id==user.id, Memory.key==key).first()
    if not m:
        m = Memory(user_id=user.id, key=key, value=value)
        db.add(m)
    else:
        m.value = value
    db.commit()

def _minutes_since(ts_iso: str | None) -> float:
    if not ts_iso: return 1e9
    try:
        dt = datetime.fromisoformat(ts_iso)
        return (datetime.utcnow() - dt).total_seconds() / 60.0
    except Exception:
        return 1e9

# ===== Llama-first classification =====
import json, re

# Allowed labels so the LLM stays consistent
SENTIMENT_LABELS = [
    "gratitude", "positive", "neutral",
    "anxiety", "stress", "depression", "suicidal", "negative"
]
EMOTION_LABELS = [
    "joy","relief","gratitude","calm","pride","love",
    "anticipation","surprise","awe",
    "fear","worry","anger","sadness","shame","guilt","disgust","confusion"
]
MOOD_LABELS = [
    "calm","confident","nervous","tense","sad","angry","tired",
    "energetic","hopeful","discouraged"
]

def _extract_json(text: str) -> dict:
    """Grab the first JSON object from a string."""
    try:
        # fast path: whole string is JSON
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}

def _norm_label(label: str, allowed: list[str]) -> str:
    if not label: return ""
    l = label.strip().lower()
    # exact or close match
    if l in allowed: return l
    # small fuzzy: map common variants
    aliases = {
        "gratified":"gratitude","grateful":"gratitude",
        "calming":"calm","stressed":"stress","worried":"worry",
        "sad":"sadness","angry":"anger","afraid":"fear",
        "shaky":"nervous","tension":"tense","neg":"negative",
    }
    l = aliases.get(l, l)
    return l if l in allowed else ""

def _norm_conf(x) -> float:
    try:
        v = float(x)
        if v > 1.0:  # percentages
            v = v/100.0
        return max(0.0, min(1.0, v))
    except Exception:
        return 0.0

def _wrap_signal(label: str, conf: float) -> dict:
    return {"label": label, "proba": conf, "probs": {label: conf} if label else {}}

LLAMA_CLASSIFIER_SYSTEM = """You are an expert emotions classifier.
Return ONLY a JSON object with keys: sentiment, emotion, mood.
Each has fields: label (string from allowed list) and confidence (0..1).
No prose. No code fencing.

Allowed:
- sentiment: """ + ", ".join(SENTIMENT_LABELS) + """
- emotion: """ + ", ".join(EMOTION_LABELS) + """
- mood: """ + ", ".join(MOOD_LABELS) + """
Guidelines:
- If the text expresses thanks/relief, sentiment must be "gratitude" or "positive", not "anxiety".
- If uncertain, choose "neutral" sentiment and "calm" or "confident" mood when appropriate.
- Confidence should reflect certainty; use values like 0.55, 0.72, 0.9 etc."""

def llama_infer_signals(text: str) -> dict:
    """Ask Llama to classify; return chips-like dict or {}."""
    try:
        resp = chat(
            model="llama3",
            messages=[
                {"role":"system","content":LLAMA_CLASSIFIER_SYSTEM},
                {"role":"user","content":f"Text: {text}\nReturn the JSON now."},
            ],
            options={"temperature":0.0}
        )
        raw = resp["message"]["content"].strip()
        data = _extract_json(raw)

        s = data.get("sentiment", {})
        e = data.get("emotion", {})
        m = data.get("mood", {})

        s_lbl = _norm_label(s.get("label",""), SENTIMENT_LABELS)
        e_lbl = _norm_label(e.get("label",""), EMOTION_LABELS)
        m_lbl = _norm_label(m.get("label",""), MOOD_LABELS)

        s_p = _norm_conf(s.get("confidence", s.get("proba", 0)))
        e_p = _norm_conf(e.get("confidence", e.get("proba", 0)))
        m_p = _norm_conf(m.get("confidence", m.get("proba", 0)))

        out = {}
        if s_lbl: out["sentiment"] = _wrap_signal(s_lbl, s_p)
        if e_lbl: out["emotion"]   = _wrap_signal(e_lbl, e_p)
        if m_lbl: out["mood"]      = _wrap_signal(m_lbl, m_p)
        return out
    except Exception as ex:
        print("[LLAMA_CLASSIFY] error:", ex)
        traceback.print_exc()
        return {}

def merge_signals(primary: dict, backup: dict, min_conf: float = 0.65) -> dict:
    """Use Llama results first, fill any missing/low-confidence with local models."""
    res = dict(primary or {})
    for t in ("sentiment","emotion","mood"):
        have = res.get(t)
        if not have or float(have.get("proba",0.0)) < min_conf:
            if t in (backup or {}):
                res[t] = backup[t]
    return res

# ---------- UI routes ----------
@app.get("/", response_class=HTMLResponse)
def root_redirect(request: Request): return templates.TemplateResponse("login.html", {"request": request})
@app.get("/ui/login", response_class=HTMLResponse)
def ui_login(request: Request): return templates.TemplateResponse("login.html", {"request": request})
@app.get("/ui/chat", response_class=HTMLResponse)
def ui_chat(request: Request): return templates.TemplateResponse("chat.html", {"request": request})

# ---------- Auth ----------
@app.post("/auth/register")
def register(inp: AuthIn, db: Session = Depends(get_db)):
    exists = db.query(User).filter(User.email == inp.email).first()
    if exists: raise HTTPException(400, "Email already registered")
    user = User(email=inp.email, password_hash=hash_password(inp.password))
    db.add(user); db.commit()
    return {"ok": True}

@app.post("/auth/login")
def login(inp: AuthIn, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == inp.email).first()
    if not user or not verify_password(inp.password, user.password_hash):
        raise HTTPException(401, "Invalid credentials")
    token = create_access_token(user.email)
    return {"access_token": token, "token_type": "bearer"}

@app.get("/me")
def me(current: User = Depends(get_current_user)):
    return {"email": current.email, "id": current.id}

# ---------- Health / Models ----------
@app.get("/health")
def health(): return {"ok": True, "tasks": available_tasks()}
@app.post("/reload")
def reload_models_endpoint(): return {"reloaded": True, "tasks": load_models()}

# ---------- Sessions ----------
@app.get("/sessions")
def list_sessions(current: User = Depends(get_current_user), db: Session = Depends(get_db)):
    sessions = db.query(ChatSession).filter(ChatSession.user_id==current.id).order_by(ChatSession.created_at.desc()).all()
    return {"items": [_session_summary(s) for s in sessions]}

@app.post("/sessions")
def create_session(current: User = Depends(get_current_user), db: Session = Depends(get_db)):
    s = ChatSession(user_id=current.id, title="New chat")
    db.add(s); db.commit(); db.refresh(s)
    return {"id": s.id, "title": s.title}

@app.get("/sessions/{sid}/messages")
def get_messages(sid: int = Path(..., gt=0), current: User = Depends(get_current_user), db: Session = Depends(get_db)):
    s = db.query(ChatSession).filter(ChatSession.id==sid, ChatSession.user_id==current.id).first()
    if not s: raise HTTPException(404, "Session not found")
    msgs = [{"id": m.id, "role": m.role, "content": m.content, "signals": m.signals, "created_at": m.created_at.isoformat()} for m in s.messages]
    return {"items": msgs}

@app.patch("/sessions/{sid}")
def rename_session(sid: int, payload: dict, current: User = Depends(get_current_user), db: Session = Depends(get_db)):
    s = db.query(ChatSession).filter(ChatSession.id==sid, ChatSession.user_id==current.id).first()
    if not s: raise HTTPException(404, "Session not found")
    title = (payload or {}).get("title","").strip()
    if title: s.title = title[:80]; db.add(s); db.commit()
    return {"id": s.id, "title": s.title}

@app.delete("/sessions/{sid}")
def delete_session(sid: int, current: User = Depends(get_current_user), db: Session = Depends(get_db)):
    s = db.query(ChatSession).filter(ChatSession.id==sid, ChatSession.user_id==current.id).first()
    if not s: raise HTTPException(404, "Session not found")
    db.delete(s); db.commit()
    return {"ok": True}

# ---------- Predict ----------
@app.post("/predict/{task}", response_model=PredictOut)
def predict(task: str, inp: PredictIn):
    bundle = get_model(task)
    if not bundle:
        raise HTTPException(404, f"Task '{task}' not found. Available: {available_tasks()}")
    try:
        X = [inp.text]
        if "hf" in bundle:
            hf = bundle["hf"]
            labels, probs = hf.predict(X)
            probs = _to_py(probs)
            # pick max safely
            best = None; best_p = -1.0
            for l, p in zip(hf.labels, probs[0]):
                if p > best_p: best, best_p = str(l), float(p)
            return PredictOut(task=task, label=str(best), proba=float(best_p))
        else:
            pipe = bundle["skl"]["pipeline"]
            label = pipe.predict(X)[0]
            proba = 0.0
            if hasattr(pipe,"predict_proba"):
                p = _to_py(pipe.predict_proba(X)[0]); proba = float(max(p))
            return PredictOut(task=task, label=str(label), proba=proba)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Prediction error: {e}")

# ---------- Unified Chat (auto therapy suggestions on negative) ----------
SYSTEM_TEMPLATE = """You are a warm, concise assistant and coach.
Continue the conversation naturally. Avoid restating the user’s detected sentiment/emotion/mood
unless it significantly changed or the user asks. Prefer short paragraphs and concrete, doable steps.

You can use the following machine signals as background context. Do not quote them unless helpful.
ML signals of the LAST user message (JSON):
{signals}

Conversation memory (recent messages + quick facts):
{memory}
"""
THERAPIST_HINT = (
    "\n\nWhen the user seems distressed, include one small, actionable exercise (2–10 minutes). "
    "Do not repeat exercises already suggested unless the user explicitly asks to review."
)

@app.post("/chat", response_model=ChatOut)
def chat_endpoint(inp: ChatIn, current: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # ensure / reuse session
    session = None
    if inp.session_id:
        session = db.query(ChatSession).filter(ChatSession.id==inp.session_id, ChatSession.user_id==current.id).first()
    if not session:
        session = ChatSession(user_id=current.id, title="Chat")
        db.add(session); db.commit(); db.refresh(session)

    # auto-title first time
    if session.title in ("New chat","Chat") and inp.text:
        tt = inp.text.strip()
        session.title = (tt[:40] + ("…" if len(tt) > 40 else ""))
        db.add(session); db.commit()

    # classify user message — Llama first, local models as backup
    llama_sig = llama_infer_signals(inp.text)
    local_sig = classify_with_local_models(inp.text)  # backup
    raw_signals = merge_signals(llama_sig, local_sig, min_conf=0.65)
    signals = _to_py(raw_signals)
    summary_now = _signals_summary(signals)

    # save user message + (optionally) chips (unchanged logic)
    prior = _mem_get(db, current, "last_signals_compact", {}) or {}
    last_task_ts = _mem_get(db, current, "last_task_suggested_at", None)

    first_turn = SHOW_CHIPS_ON_FIRST and not prior
    chips_change = _signals_changed(prior, summary_now)
    show_chips = bool(first_turn or chips_change)

    # decide whether to include therapist tasks (on negative transition or cooldown passed)
    risky = detect_risk(inp.text, signals)
    negative_now = is_negative(signals, inp.text)
    negative_then = any((prior.get("sentiment") or {}).get("label", "") in {"anxiety","stress","depression","suicidal"}
                        for _ in [0]) or any((prior.get(k) or {}).get("label","") in {"sad","fear","anger"} for k in ("emotion","mood"))
    minutes_since_task = _minutes_since(last_task_ts)
    allow_task = negative_now and (not negative_then or minutes_since_task >= TASK_COOLDOWN_MIN)

    tasks = suggest_tasks(signals, inp.text) if allow_task else []

    # save user message + (optionally) chips
    store_signals = signals if show_chips else None
    db.add(Message(session_id=session.id, role="user", content=inp.text, signals=store_signals)); db.commit()

    # build memory + system prompt
    memory_text = build_memory_summary(db, current, k=20)
    sys_prompt = SYSTEM_TEMPLATE.format(signals=signals if show_chips else {}, memory=memory_text)
    if allow_task:
        sys_prompt += THERAPIST_HINT

    # ask LLM
    llm = chat(model="llama3", messages=[
        {"role":"system","content":sys_prompt},
        {"role":"user","content":inp.text}
    ], options={"temperature":0.3})
    reply = llm["message"]["content"].strip()

    # save assistant message (we don't attach chips to assistant turns)
    db.add(Message(session_id=session.id, role="assistant", content=reply, signals=None)); db.commit()

    # update rolling memories
    update_rolling_memory(db, current, signals)  # last_* labels
    _mem_set(db, current, "last_signals_compact", summary_now)
    if tasks:
        _mem_set(db, current, "last_task_suggested_at", datetime.utcnow().isoformat())

    # respond to UI: chips appear only when show_chips is True
    return ChatOut(
        session_id=session.id,
        reply=reply,
        signals=signals if show_chips else {},  # <-- chips hidden unless change
        tasks=tasks if tasks else None,
        risky=risky
    )