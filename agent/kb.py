"""Tiny RAG knowledge base: policy documents plus a pure-Python TF-IDF retriever.

Good enough for a demo and easy to read. To upgrade, replace `KnowledgeBase.search`
with embeddings (e.g. sentence-transformers) and a vector store (Chroma, FAISS).
"""
from __future__ import annotations

import math
import re
from collections import Counter

DOCS = [
    {"title": "Why is my bill higher than usual",
     "text": "A bill can rise because of one-time charges such as international roaming packs, "
             "data add-ons, late payment fees, or a plan change that took effect this cycle. "
             "Ask the agent for the bill breakdown to see each line. GST of 18 percent is added to all charges."},
    {"title": "Late payment fee",
     "text": "If a postpaid bill is not paid by the due date, a late fee of 2 percent of the bill "
             "amount, minimum 50 rupees, is charged. The fee is waived once per year on request "
             "if the customer has no earlier late payments."},
    {"title": "International roaming",
     "text": "International roaming must be activated before travel using a roaming pack. "
             "The 7 day pack costs 599 rupees and the 30 day pack costs 1799 rupees. "
             "Without a pack, roaming is blocked. Activation takes up to 15 minutes. "
             "Postpaid customers only need a clear bill history to activate roaming."},
    {"title": "No signal or internet not working",
     "text": "First check for a known outage in the area. If there is none, ask the customer to switch "
             "airplane mode on for 10 seconds, restart the phone, and check that mobile data and the "
             "correct APN are enabled. If the problem continues after these steps, raise a network ticket. "
             "Network tickets are resolved within 24 hours."},
    {"title": "Changing plans",
     "text": "Prepaid plan changes apply immediately, after the current pack expires or as a new recharge. "
             "Postpaid plan changes take effect from the next bill cycle. A customer cannot move "
             "between prepaid and postpaid using the voice agent; this needs a store visit or a human agent."},
    {"title": "Data add-on packs",
     "text": "Data add-ons give extra data on top of the plan. The 1 GB pack costs 19 rupees for 1 day, "
             "the 2 GB pack costs 29 rupees for 2 days, and the 5 GB pack costs 98 rupees for 7 days. "
             "For prepaid customers the cost is deducted from the wallet. For postpaid it is added to the next bill."},
    {"title": "Refunds and billing disputes",
     "text": "Refunds for wrong charges are handled by a human agent after a billing dispute ticket is raised. "
             "The agent cannot promise refunds. Disputes are reviewed within 5 working days."},
    {"title": "SIM lost or stolen",
     "text": "For a lost or stolen SIM, the number should be blocked immediately. A human agent must "
             "handle this and a replacement SIM is issued at a store with a photo ID."},
    {"title": "Number portability",
     "text": "Port out requests need a unique porting code sent by SMS to the registered number. "
             "Port out is handled by a human agent. The agent should not try to retain the customer by inventing offers."},
    {"title": "5G service",
     "text": "5G is included at no extra cost on Postpaid 599 and above. It needs a 5G capable phone and "
             "coverage in the area. Other plans can use 5G at standard speeds where available."},
    {"title": "When to transfer to a human",
     "text": "Transfer to a human when the customer asks for one, is very upset, wants a refund, "
             "reports a lost SIM, wants to port out, or when the agent cannot solve the issue after two attempts."},
]

_STOP = {"the", "a", "an", "is", "are", "to", "of", "and", "or", "in", "on", "for", "my", "me", "i", "it",
         "how", "do", "does", "can", "what", "why", "you", "be", "this", "that", "with", "at", "by", "if"}


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words if w not in _STOP]


class KnowledgeBase:
    def __init__(self, docs: list[dict] | None = None):
        self.docs = docs or DOCS
        self._tf = [Counter(_tokens(d["title"] + " " + d["title"] + " " + d["text"])) for d in self.docs]
        df: Counter = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(self.docs)
        self._idf = {w: math.log((n + 1) / (c + 0.5)) + 1 for w, c in df.items()}
        self._vecs = [self._vec(tf) for tf in self._tf]

    def _vec(self, tf: Counter) -> dict[str, float]:
        v = {w: c * self._idf.get(w, 0.0) for w, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {w: x / norm for w, x in v.items()}

    def search(self, query: str, k: int = 2, min_score: float = 0.08) -> list[dict]:
        q = self._vec(Counter(_tokens(query)))
        scored = []
        for doc, vec in zip(self.docs, self._vecs):
            score = sum(x * vec.get(w, 0.0) for w, x in q.items())
            if score >= min_score:
                scored.append((score, doc))
        scored.sort(key=lambda s: s[0], reverse=True)
        return [{"title": d["title"], "text": d["text"], "score": round(s, 3)} for s, d in scored[:k]]
