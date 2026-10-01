# Telecom Voice Agent

A voice agent that handles common telecom support calls: data usage, bill explanations, plan changes,
data and roaming packs, network problems, and policy questions. It verifies the caller, asks for
confirmation before changing anything, raises tickets, and hands over to a human with a summary.

```
Browser mic → speech-to-text → /api/chat → LLM + tools → reply text → speech → speaker
                                              │
                          ┌───────────────────┼─────────────────────┐
                          ▼                   ▼                     ▼
                   backend/store.py     agent/kb.py           tickets / transfer
                   (mock telecom systems) (RAG over policies)
```

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env            # set OPENAI_API_KEY or ANTHROPIC_API_KEY; leave both empty for offline demo mode
uvicorn server:app --reload     # open http://localhost:8000 in Chrome or Edge, click "Start call"
```

Text only, in the terminal: `python cli.py --trace`
Evals: `python -m evals.run_evals`

**Offline demo mode** (no API key) uses `agent/mock_llm.py`, a keyword-rule stand-in. It lets you see the whole
flow, but it is not a real LLM. Set `OPENAI_API_KEY` (OpenAI) or `ANTHROPIC_API_KEY` (Claude) to use a real model.
If both are set, Anthropic wins unless you set `LLM_PROVIDER=openai`. Change the model with `OPENAI_MODEL` or `ANTHROPIC_MODEL`.

Demo logins (fake data): `9876543210 / 1234` postpaid, roaming charge on the bill, tower outage in the area;
`9123456780 / 4321` prepaid, no outage; `9988776655 / 2468` postpaid.

Try: "Why is my bill so high?", "Activate a 5GB pack", "My internet is not working", "Change my plan",
"What is the late fee?", "I want to talk to a human".

## Project layout

| Path | What it does |
|---|---|
| `backend/store.py` | Mock telecom systems: customers, plans, add-ons, outages, tickets |
| `agent/tools.py` | Tool schemas, session state, and the guardrails enforced in code |
| `agent/agent.py` | System prompt (voice style rules) and the tool-calling loop |
| `agent/llm.py` | Anthropic and OpenAI adapters, and provider selection |
| `agent/mock_llm.py` | Offline rule-based stand-in for testing without a key |
| `agent/kb.py` | Policy documents and a small TF-IDF retriever |
| `server.py`, `static/index.html` | Web server and browser call console (Web Speech API) |
| `evals/run_evals.py` | 7 guardrail checks and 12 scenario tests |

## Guardrails (enforced in code, not only in the prompt)

1. Account tools take no phone number, so the model can only act on the verified caller.
2. Identity check (number and PIN) comes first. Three wrong PINs lock the session.
3. Changing a plan or activating a pack is two steps. The first call only returns a read-back. The second works
   only if the caller answered in a later turn, and only for the same item. Confirmations cannot be replayed.
4. Tool errors never crash the call. The model gets the error and recovers.
5. PINs are masked in the trace.

## Voice notes

- **Current voice layer:** the browser does speech-to-text and text-to-speech. It is free and needs no setup, but
  quality varies by browser and there is no true voice-activity barge-in. Tap the mic to interrupt.
  Use headphones, otherwise the mic hears the agent and gets confused.
- **Latency:** the UI shows agent time and round-trip time per turn. The target is under about 1 second from the
  caller finishing to the agent starting. Keep replies to 1 or 2 sentences and use a fast model.
- **Speech recognition errors:** phone numbers and amounts get misheard, so the prompt makes the agent read
  important details back before acting.

## Next steps

1. **Real-time streaming voice.** Move to LiveKit Agents or Pipecat with Deepgram or Whisper for speech-to-text and
   Cartesia or ElevenLabs for text-to-speech. Reuse `agent/tools.py`, `agent/kb.py` and `SYSTEM_PROMPT` as they are.
   The framework adds voice-activity detection, streaming, and real barge-in.
2. **Real telephony.** Connect Twilio or a SIP trunk through LiveKit or Pipecat.
3. **Better retrieval.** Replace TF-IDF in `agent/kb.py` with embeddings and a vector store (Chroma, FAISS).
4. **Per-call state.** The mock backend is one shared in-memory store, fine for a demo. Use a database for real use.
5. **More evals.** Grow to 30 to 50 scenarios: angry caller, unclear audio, caller changes their mind, prompt
   injection attempts. Track task success, tool-call accuracy and latency per model.
6. **Observability.** Log transcripts and tool traces, and add a dashboard for outcomes and escalation rate.
