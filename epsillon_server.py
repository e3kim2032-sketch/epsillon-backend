# epsillon_server.py
# Epsillon backend — deployable to Render.com
# Uses Google Gemini API (free tier works!)
# Run locally: uvicorn epsillon_server:app --host 0.0.0.0 --port 8000

from __future__ import annotations
import os
import re
import json
import base64
import tempfile
from datetime import date, datetime
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai

# ─── Load .env ────────────────────────────────────────────────────────────

def _load_env():
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v

_load_env()

# ─── Gemini client ────────────────────────────────────────────────────────

GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
genai.configure(api_key=GEMINI_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

# ─── Storage paths ────────────────────────────────────────────────────────

DATA_DIR     = Path(os.getenv("EPSILLON_DATA_DIR", "/tmp/epsillon_data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
MEMORY_FILE  = DATA_DIR / "memory.json"
PROFILE_FILE = DATA_DIR / "user_profile.json"
DRIFT_FILE   = DATA_DIR / "drift_tasks.json"
DREAM_FILE   = DATA_DIR / "dream_log.json"

# ─── Helpers ──────────────────────────────────────────────────────────────

def load_json(path: Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return default

def save_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

def load_profile() -> str:
    d = load_json(PROFILE_FILE, {})
    return d.get("profile", "User profile not set.")

def load_memory() -> list:
    d = load_json(MEMORY_FILE, [])
    return d if isinstance(d, list) else []

def save_memory(m: list): save_json(MEMORY_FILE, m)

def load_drift() -> list:
    d = load_json(DRIFT_FILE, [])
    return d if isinstance(d, list) else []

def save_drift(t: list): save_json(DRIFT_FILE, t)

def load_dream() -> list:
    d = load_json(DREAM_FILE, [])
    return d if isinstance(d, list) else []

def save_dream(e: list): save_json(DREAM_FILE, e)

# ─── System prompt ────────────────────────────────────────────────────────

def build_system_prompt() -> str:
    today = date.today().strftime("%A, %B %d, %Y")
    profile = load_profile()
    memory = load_memory()
    tasks = load_drift()

    mem_text = f"Long-term memory: {'; '.join(memory[-10:])}." if memory else ""
    pending = [t for t in tasks if not t.get("done")]
    drift_text = f"Pending tasks: {', '.join(t['task'] for t in pending[:5])}." if pending else ""

    return (
        f"You are Epsillon, a private AI assistant living in smart glasses. "
        f"Today is {today}. User profile: {profile}. {mem_text} {drift_text} "
        "Keep responses short — max 2 sentences — for OLED display. "
        "Be direct, smart, and helpful. Never mention being an AI unless asked. "
        "If the user says something worth remembering, end with: [[REMEMBER: short fact]]"
    )

# ─── App ──────────────────────────────────────────────────────────────────

app = FastAPI(title="Epsillon Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory conversation history (resets on server restart — use Render persistent disk for prod)
conversation_history: list[dict] = []

# ─── Routes ───────────────────────────────────────────────────────────────

@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {"status": "Epsillon backend running", "model": "gemini-2.0-flash"}

@app.post("/epsillon/text")
async def handle_text(body: dict):
    text = body.get("message", "").strip()
    if not text:
        return JSONResponse({"response": "Empty message."})
    return JSONResponse({"response": _chat(text)})

@app.post("/epsillon/audio")
async def handle_audio(audio: UploadFile = File(...)):
    """Receives audio, transcribes with Gemini, returns AI response."""
    try:
        audio_bytes = await audio.read()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # Use Gemini for STT
        audio_file = genai.upload_file(tmp_path, mime_type="audio/wav")
        transcript_response = model.generate_content([
            "Transcribe this audio exactly. Return only the transcribed text, nothing else.",
            audio_file
        ])
        os.unlink(tmp_path)

        user_text = transcript_response.text.strip()
        if not user_text:
            return JSONResponse({"response": "I didn't catch that, try again."})

        reply = _chat(user_text)
        return JSONResponse({"response": reply, "transcript": user_text})

    except Exception as e:
        return JSONResponse({"response": f"Error: {e}"}, status_code=500)

@app.post("/epsillon/image")
async def handle_image(image: UploadFile = File(...)):
    """Vision query — describe what the glasses see."""
    try:
        image_bytes = await image.read()
        b64 = base64.b64encode(image_bytes).decode()
        response = model.generate_content([
            {"mime_type": "image/jpeg", "data": b64},
            "Describe what you see in one short sentence. Focus on setting and activity."
        ])
        return JSONResponse({"response": response.text.strip()})
    except Exception as e:
        return JSONResponse({"response": f"Error: {e}"}, status_code=500)

@app.post("/epsillon/drift")
async def handle_drift(body: dict):
    """Extract tasks from free text."""
    text = body.get("text", "").strip()
    if not text:
        return JSONResponse({"error": "No text provided."}, status_code=400)
    try:
        response = model.generate_content(
            f'Extract all tasks and action items from this text. '
            f'Return ONLY a JSON array of short strings. '
            f'Example: ["Call Jake", "Buy groceries"]\n\nText: {text}'
        )
        raw = response.text.strip().replace("```json", "").replace("```", "").strip()
        extracted = json.loads(raw)
        if not isinstance(extracted, list):
            extracted = []
        tasks = load_drift()
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        for task in extracted:
            tasks.append({"task": task, "added": now, "done": False})
        save_drift(tasks)
        return JSONResponse({
            "extracted": extracted,
            "total_pending": len([t for t in tasks if not t["done"]]),
            "message": f"Drift found {len(extracted)} task(s)."
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/epsillon/drift/tasks")
def get_drift_tasks():
    tasks = load_drift()
    pending = [t for t in tasks if not t.get("done")]
    return JSONResponse({"tasks": pending, "count": len(pending)})

@app.post("/epsillon/dream")
async def handle_dream(body: dict):
    """Generate a private journal entry from the day."""
    extra_notes = body.get("notes", "").strip()
    try:
        today = date.today().strftime("%A, %B %d, %Y")
        memory = load_memory()
        tasks = load_drift()

        mem_ctx = f"Things remembered: {'; '.join(memory[-15:])}." if memory else ""
        done = [t["task"] for t in tasks if t.get("done")]
        pending = [t["task"] for t in tasks if not t.get("done")]
        task_ctx = ""
        if done: task_ctx += f"Completed: {', '.join(done)}. "
        if pending: task_ctx += f"Still pending: {', '.join(pending)}."

        conv_ctx = "\n".join(
            f"{'You' if m['role'] == 'user' else 'Epsillon'}: {m['content']}"
            for m in conversation_history[-20:]
            if isinstance(m.get("content"), str)
        )

        prompt = (
            f"Write a private personal journal entry for {today}. "
            f"First person, warm and reflective. 3-5 sentences. "
            f"No mention of AI or technology.\n\n"
            f"{mem_ctx}\n{task_ctx}\n"
            f"{'Notes: ' + extra_notes if extra_notes else ''}\n"
            f"Conversation highlights:\n{conv_ctx}"
        )

        response = model.generate_content(prompt)
        entry_text = response.text.strip()

        log = load_dream()
        log.append({
            "date": today,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "entry": entry_text,
            "private": True
        })
        save_dream(log)

        return JSONResponse({
            "entry": entry_text,
            "saved": True,
            "note": "Stored on Render persistent disk only."
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/epsillon/dream/log")
def get_dream_log():
    return JSONResponse({"entries": load_dream(), "count": len(load_dream())})

# ─── WebSocket (for real-time glasses connection) ─────────────────────────

@app.websocket("/ws/epsillon")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            text = msg.get("text", "").strip()
            if text:
                reply = _chat(text)
                await websocket.send_text(json.dumps({"response": reply}))
    except WebSocketDisconnect:
        pass

# ─── LLM helper ───────────────────────────────────────────────────────────

def _chat(user_text: str) -> str:
    global conversation_history
    conversation_history.append({"role": "user", "content": user_text})

    # Build Gemini chat with history
    history = []
    for msg in conversation_history[-20:]:
        role = "user" if msg["role"] == "user" else "model"
        history.append({"role": role, "parts": [msg["content"]]})

    system = build_system_prompt()
    chat_session = model.start_chat(history=history[:-1])
    response = chat_session.send_message(
        f"[SYSTEM: {system}]\n\n{user_text}"
    )

    reply = response.text.strip()
    conversation_history.append({"role": "assistant", "content": reply})

    if "[[REMEMBER:" in reply:
        fact = reply.split("[[REMEMBER:")[1].split("]]")[0].strip()
        mem = load_memory()
        mem.append(fact)
        save_memory(mem)
        reply = re.sub(r'\[\[REMEMBER:.*?\]\]', '', reply).strip()

    return reply
