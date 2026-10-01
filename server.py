"""Web server. Run:  uvicorn server:app --reload   then open http://localhost:8000

The browser handles the microphone and speaker (Web Speech API); this server is the
agent brain. To move to real-time streaming voice, keep agent/ and backend/ as they
are and swap this file for a LiveKit Agents or Pipecat worker (see README).
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent.agent import GREETING, VoiceAgent
from agent.llm import make_llm
from backend import store

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).parent
app = FastAPI(title="Telecom Voice Agent")
SESSIONS: dict[str, VoiceAgent] = {}


class ChatIn(BaseModel):
    session_id: str
    text: str


class SessionIn(BaseModel):
    session_id: str


@app.get("/api/info")
def info():
    llm = make_llm()
    return {"llm": llm.name, "label": llm.label, "greeting": GREETING,
            "demo_customers": [{"phone": p, "pin": c["pin"], "name": c["name"], "type": c["type"]}
                               for p, c in store.CUSTOMERS.items()]}


@app.post("/api/reset")
def reset(body: SessionIn):
    """Start a fresh call. Also restores the mock backend so demos are repeatable."""
    store.reset()
    SESSIONS[body.session_id] = VoiceAgent()
    return {"greeting": GREETING, "state": SESSIONS[body.session_id].session.public_state()}


@app.post("/api/chat")
def chat(body: ChatIn):
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "Empty message")
    agent = SESSIONS.setdefault(body.session_id, VoiceAgent())
    return agent.respond(text)


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
