"""Agent tools and guardrails.

Guardrails live in CODE, not just in the prompt:
  1. Account tools take no phone number. They act on the verified caller only,
     so the model cannot be talked into reading someone else's account.
  2. Identity check (phone + 4 digit PIN) is required first; 3 wrong tries lock the session.
  3. State-changing tools need a two step confirmation: the first call only returns a
     read-back, and the second call (confirmed=true) works only if the caller replied
     in a LATER turn.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend import store
from agent.kb import KnowledgeBase

MAX_PIN_ATTEMPTS = 3
KB = KnowledgeBase()


@dataclass
class Session:
    verified_phone: str | None = None
    failed_attempts: int = 0
    locked: bool = False
    turn: int = 0
    pending: dict | None = None
    transferred: bool = False
    transfer_info: dict | None = None
    tickets: list[str] = field(default_factory=list)

    def public_state(self) -> dict:
        name = store.CUSTOMERS[self.verified_phone]["name"] if self.verified_phone else None
        return {"verified": bool(self.verified_phone), "customer": name, "locked": self.locked,
                "transferred": self.transferred, "tickets": self.tickets,
                "pending_confirmation": bool(self.pending)}


TOOLS: list[dict[str, Any]] = [
    {"name": "verify_customer",
     "description": "Verify the caller's identity with their 10 digit mobile number and 4 digit PIN. "
                    "Must succeed before any account tool works.",
     "input_schema": {"type": "object", "properties": {
         "phone_number": {"type": "string", "description": "10 digit mobile number, digits only"},
         "pin": {"type": "string", "description": "4 digit security PIN"}},
         "required": ["phone_number", "pin"]}},
    {"name": "get_usage",
     "description": "Data used, data left, plan name, cycle end date (and wallet balance for prepaid) for the verified caller.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "get_bill_summary",
     "description": "Latest bill with each line item and the change from the previous bill (postpaid). Use for 'why is my bill high'.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "list_plans",
     "description": "Available plans (filtered to the caller's type once verified).",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "change_plan",
     "description": "Change the caller's plan. First call with confirmed=false to get a read-back, "
                    "read it to the caller, and only after they say yes call again with confirmed=true.",
     "input_schema": {"type": "object", "properties": {
         "plan_id": {"type": "string"}, "confirmed": {"type": "boolean"}},
         "required": ["plan_id", "confirmed"]}},
    {"name": "list_addons",
     "description": "Data packs and roaming packs with prices and validity.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "activate_addon",
     "description": "Activate a data or roaming pack. Same two step confirmation as change_plan.",
     "input_schema": {"type": "object", "properties": {
         "addon_id": {"type": "string"}, "confirmed": {"type": "boolean"}},
         "required": ["addon_id", "confirmed"]}},
    {"name": "run_network_diagnostic",
     "description": "Check for a known outage and signal quality at the caller's registered area.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "create_ticket",
     "description": "Raise a support ticket (categories: network, billing_dispute, other).",
     "input_schema": {"type": "object", "properties": {
         "category": {"type": "string", "enum": ["network", "billing_dispute", "other"]},
         "description": {"type": "string"}},
         "required": ["category", "description"]}},
    {"name": "transfer_to_human",
     "description": "Hand the call to a human agent with a short summary so the caller does not repeat themselves. "
                    "Works without verification.",
     "input_schema": {"type": "object", "properties": {
         "reason": {"type": "string"}, "summary": {"type": "string"}},
         "required": ["reason", "summary"]}},
    {"name": "search_knowledge",
     "description": "Search company policies and FAQs. Use for any policy or how-it-works question instead of guessing.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
]

NEEDS_VERIFICATION = {"get_usage", "get_bill_summary", "change_plan", "activate_addon",
                      "run_network_diagnostic", "create_ticket"}


def _digits(s: Any) -> str:
    return "".join(ch for ch in str(s) if ch.isdigit())


def _confirmation_gate(session: Session, key: dict, read_back: str, confirmed: bool) -> dict | None:
    """Return a dict to send back to the model, or None if the action may proceed."""
    if not confirmed:
        session.pending = {"key": key, "turn": session.turn}
        return {"status": "confirmation_required", "read_back": read_back,
                "instruction": "Read this to the caller and ask for a yes. Do not act yet."}
    p = session.pending
    if not p or p["key"] != key:
        return {"error": "No matching confirmation request. Call with confirmed=false first."}
    if session.turn <= p["turn"]:
        return {"error": "The caller has not answered yet. Ask them and wait for their reply."}
    session.pending = None
    return None


def run_tool(session: Session, name: str, args: dict) -> dict:
    try:
        return _dispatch(session, name, args or {})
    except Exception as exc:  # never crash a live call because of a tool bug
        return {"error": f"Tool failed: {exc}"}


def _dispatch(session: Session, name: str, args: dict) -> dict:
    if name in NEEDS_VERIFICATION and not session.verified_phone:
        return {"error": "Caller is not verified. Ask for their mobile number and PIN first."}

    phone = session.verified_phone

    if name == "verify_customer":
        if session.locked:
            return {"verified": False, "locked": True,
                    "message": "Too many failed attempts. Offer a human agent."}
        number, pin = _digits(args.get("phone_number")), _digits(args.get("pin"))
        if store.customer_exists(number) and store.check_pin(number, pin):
            session.verified_phone, session.failed_attempts = number, 0
            return {"verified": True, "name": store.CUSTOMERS[number]["name"],
                    "account": store.get_account(number)}
        session.failed_attempts += 1
        if session.failed_attempts >= MAX_PIN_ATTEMPTS:
            session.locked = True
            return {"verified": False, "locked": True,
                    "message": "Locked after 3 failed attempts. Offer a human agent."}
        return {"verified": False, "attempts_left": MAX_PIN_ATTEMPTS - session.failed_attempts}

    if name == "get_usage":
        return store.get_usage(phone)
    if name == "get_bill_summary":
        return store.get_bill(phone)
    if name == "list_plans":
        ctype = store.CUSTOMERS[phone]["type"] if phone else None
        return {"plans": store.list_plans(ctype)}
    if name == "list_addons":
        return {"addons": store.list_addons()}

    if name == "change_plan":
        plan_id = args.get("plan_id", "")
        plan = store.PLANS.get(plan_id)
        if not plan:
            return {"error": f"Unknown plan {plan_id}"}
        gate = _confirmation_gate(
            session, {"action": name, "plan_id": plan_id},
            f"Change to {plan['name']} at {plan['price']} rupees, {plan['data']}.", bool(args.get("confirmed")))
        return gate or store.change_plan(phone, plan_id)

    if name == "activate_addon":
        addon_id = args.get("addon_id", "")
        addon = store.ADDONS.get(addon_id)
        if not addon:
            return {"error": f"Unknown add-on {addon_id}"}
        gate = _confirmation_gate(
            session, {"action": name, "addon_id": addon_id},
            f"Activate {addon['name']} for {addon['price']} rupees, valid {addon['validity']}.",
            bool(args.get("confirmed")))
        return gate or store.activate_addon(phone, addon_id)

    if name == "run_network_diagnostic":
        return store.network_status(phone)

    if name == "create_ticket":
        result = store.create_ticket(phone, args.get("category", "other"), args.get("description", ""))
        session.tickets.append(result["ticket_id"])
        return result

    if name == "transfer_to_human":
        session.transferred = True
        session.transfer_info = {"reason": args.get("reason", ""), "summary": args.get("summary", ""),
                                 "verified": bool(phone)}
        return {"status": "transferring", "wait_time": "about 2 minutes"}

    if name == "search_knowledge":
        hits = KB.search(args.get("query", ""))
        return {"results": hits} if hits else {"results": [], "note": "Nothing relevant found. Do not guess."}

    return {"error": f"Unknown tool {name}"}
