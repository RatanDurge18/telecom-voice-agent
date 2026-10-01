"""Offline stand-in for the LLM.

It is NOT smart. It uses keyword rules so the UI, tools, guardrails and evals can run
without an API key. It speaks the same interface as AnthropicLLM, so the agent loop
is identical. Set ANTHROPIC_API_KEY to use a real model.
"""
from __future__ import annotations

import json
import re

from agent.llm import LLMResponse
from backend import store

HUMAN_WORDS = ["human", "agent", "representative", "real person", "customer care", "executive",
               "refund", "lost sim", "sim is lost", "stolen", "port out", "port my number"]
YES = re.compile(r"\b(yes|yeah|yep|sure|confirm|go ahead|correct|do it|please do|okay|ok)\b")
NO = re.compile(r"\b(no|nope|cancel|don't|do not|stop|wait)\b")
TELECOM_WORDS = ["sim", "plan", "roaming", "recharge", "bill", "data", "tower", "5g", "port", "fee",
                 "charge", "validity", "pack", "network", "signal", "internet"]
NEEDS_VERIFICATION = {"usage", "bill", "network", "addon", "plan_change", "plan_pick"}


class MockLLM:
    name = "mock"
    label = "Offline demo mode (keyword rules, not a real LLM)"

    def __init__(self):
        self.phone: str | None = None
        self.pin: str | None = None
        self.verified = False
        self.locked = False
        self.goal: dict | None = None
        self.awaiting: tuple[str, dict] | None = None
        self._n = 0

    # ---------- interface ----------
    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        last = messages[-1]
        if isinstance(last["content"], list):
            return self._after_tool(messages)
        return self._on_user(last["content"])

    # ---------- helpers ----------
    def _say(self, text: str) -> LLMResponse:
        return LLMResponse(text=text)

    def _call(self, name: str, **kwargs) -> LLMResponse:
        self._n += 1
        return LLMResponse(tool_calls=[{"id": f"mock_{self._n}", "name": name, "input": kwargs}])

    @staticmethod
    def _creds(t: str):
        joined = re.sub(r"(?<=\d)[\s-]+(?=\d)", "", t)
        phone = pin = None
        for run in re.findall(r"\d+", joined):
            if len(run) == 14:
                phone, pin = run[:10], run[10:]
            elif len(run) == 10:
                phone = run
            elif len(run) == 4:
                pin = run
        return phone, pin

    @staticmethod
    def _plan_in(t: str) -> str | None:
        for pid, p in store.PLANS.items():
            if str(p["price"]) in t:
                return pid
        return None

    def _intent(self, t: str) -> dict | None:
        has = lambda *ws: any(w in t for w in ws)
        if "roaming" in t and has("activate", "enable", "turn on", "start", "need", "want", "add"):
            return {"intent": "addon", "addon_id": "ROAM_INTL_30D" if "30" in t else "ROAM_INTL_7D"}
        gb = re.search(r"(\d+)\s*(gb|g b|gigabyte)", t)
        if gb and has("pack", "add", "activate", "recharge", "top up", "extra") and gb.group(1) in ("1", "2", "5"):
            return {"intent": "addon", "addon_id": f"DATA_{gb.group(1)}GB"}
        if has("usage", "data left", "how much data", "data used", "remaining", "balance"):
            return {"intent": "usage"}
        if "bill" in t and not has("late fee", "policy"):
            return {"intent": "bill"}
        if has("network", "signal", "internet", "no service", "not working", "slow", "call drop"):
            return {"intent": "network"}
        if "plan" in t:
            if has("change", "switch", "upgrade", "downgrade", "move to"):
                target = self._plan_in(t)
                return {"intent": "plan_change", "plan_id": target} if target else {"intent": "plan_pick"}
            return {"intent": "plan_info"}
        if has("pack", "ran out", "recharge", "top up"):
            return {"intent": "addon_choose"}
        if has(*TELECOM_WORDS):
            return {"intent": "knowledge", "query": t}
        return None

    # ---------- user turn ----------
    def _on_user(self, text: str) -> LLMResponse:
        t = text.lower().strip()
        phone, pin = self._creds(t)
        self.phone = phone or self.phone
        if pin:
            self.pin = pin

        if self.awaiting:
            if YES.search(t):
                name, args = self.awaiting
                self.awaiting = None
                return self._call(name, **args, confirmed=True)
            if NO.search(t):
                self.awaiting, self.goal = None, None
                return self._say("Okay, I have not made any change. Is there anything else I can help with?")

        if any(w in t for w in HUMAN_WORDS):
            self.goal = None
            return self._call("transfer_to_human", reason="Caller asked for a human or has a case the agent cannot handle",
                              summary=f"Caller said: {text}")

        if self.goal and self.goal["intent"] == "plan_pick" and self._plan_in(t):
            self.goal = {"intent": "plan_change", "plan_id": self._plan_in(t)}
        else:
            new_goal = self._intent(t)
            if new_goal:
                self.goal = new_goal

        if self.phone and self.pin and not self.verified and not self.locked:
            return self._call("verify_customer", phone_number=self.phone, pin=self.pin)

        if not self.goal:
            if re.match(r"^(hi|hello|hey|good (morning|afternoon|evening))\b", t):
                return self._say("Hi! How can I help you today?")
            if self.verified and (phone or pin):
                return self._say("Thanks. What can I help you with?")
            return self._say("Sorry, I can only help with your telecom service, like data usage, bills, plans, "
                             "packs and network issues. What do you need?")
        return self._advance()

    def _advance(self) -> LLMResponse:
        g = self.goal
        intent = g["intent"]
        if intent in NEEDS_VERIFICATION and not self.verified:
            if self.locked:
                return self._say("For your security I cannot verify you right now. I can transfer you to a human agent.")
            if self.phone and not self.pin:
                return self._say("Thanks. Please tell me your 4 digit PIN.")
            return self._say("Sure, I can help with that. First, please tell me your 10 digit mobile number and 4 digit PIN.")
        if intent == "usage":
            return self._call("get_usage")
        if intent == "bill":
            return self._call("get_bill_summary")
        if intent == "network":
            return self._call("run_network_diagnostic")
        if intent == "addon":
            return self._call("activate_addon", addon_id=g["addon_id"], confirmed=False)
        if intent == "addon_choose":
            return self._say("Which data pack would you like, 1 GB, 2 GB or 5 GB?")
        if intent == "plan_change":
            return self._call("change_plan", plan_id=g["plan_id"], confirmed=False)
        if intent in ("plan_pick", "plan_info"):
            return self._call("list_plans")
        if intent == "knowledge":
            return self._call("search_knowledge", query=g["query"])
        return self._say("How can I help?")

    # ---------- after a tool result ----------
    def _after_tool(self, messages: list[dict]) -> LLMResponse:
        result_block = messages[-1]["content"][-1]
        out = json.loads(result_block["content"])
        call = next(b for b in messages[-2]["content"]
                    if b.get("type") == "tool_use" and b["id"] == result_block["tool_use_id"])
        name, args = call["name"], call["input"]

        if "error" in out:
            self.goal = None
            return self._say(out["error"] if name != "activate_addon" or "Insufficient" not in out["error"]
                             else f"Your wallet balance is {out['wallet_balance_rupees']:g} rupees, which is not enough for this pack.")

        if name == "verify_customer":
            if out.get("verified"):
                self.verified = True
                if self.goal:
                    return self._advance()
                return self._say(f"Thank you {out['name'].split()[0]}, you are verified. How can I help?")
            self.pin = None
            if out.get("locked"):
                self.locked = True
                return self._say("Sorry, that did not match and I have locked verification for this call. I can transfer you to a human agent.")
            return self._say("Sorry, that did not match. Please say your mobile number and PIN again.")

        if name == "get_usage":
            self.goal = None
            msg = (f"You have used {out['data_used_gb']:g} of {out['data_total_gb']:g} GB, "
                   f"so {out['data_left_gb']:g} GB is left until {out['cycle_end']}.")
            if "wallet_balance_rupees" in out:
                msg += f" Your wallet balance is {out['wallet_balance_rupees']:g} rupees."
            return self._say(msg)

        if name == "get_bill_summary":
            self.goal = None
            if out["type"] == "prepaid":
                return self._say(out["note"])
            extras = [l for l in out["lines"] if "plan" not in l["item"].lower() and "gst" not in l["item"].lower()]
            msg = f"Your bill is {out['total']:g} rupees, due on {out['due_date']}."
            if out["difference_from_previous"] > 1 and extras:
                top = max(extras, key=lambda l: l["amount"])
                msg = (f"Your bill is {out['total']:g} rupees, which is {out['difference_from_previous']:g} more than last month. "
                       f"The main reason is {top['item'].split(' (')[0]} for {top['amount']:g} rupees.")
            return self._say(msg)

        if name == "run_network_diagnostic":
            if out["outage"]:
                self.goal = None
                return self._say(out["message"] + " I am sorry for the trouble.")
            return self._call("create_ticket", category="network",
                              description="Caller reports no or poor service. Diagnostic found no outage.")

        if name == "create_ticket":
            self.goal = None
            return self._say(f"I found no outage, so I have raised ticket {out['ticket_id']}, resolved {out['eta']}. "
                             "Meanwhile, turn airplane mode on for ten seconds and then restart your phone.")

        if name == "list_plans":
            plans = out["plans"]
            spoken = "; ".join(f"{p['name']} for {p['price']} rupees" for p in plans[:4])
            if self.goal and self.goal["intent"] == "plan_info":
                self.goal = None
                return self._say(f"We have {spoken}.")
            return self._say(f"Your options are {spoken}. Which one would you like?")

        if name in ("change_plan", "activate_addon"):
            if out.get("status") == "confirmation_required":
                self.awaiting = (name, {k: v for k, v in args.items() if k != "confirmed"})
                return self._say(out["read_back"] + " Shall I go ahead?")
            self.goal = None
            if name == "change_plan":
                return self._say(f"Done. You are now on {out['new_plan']}, effective {out['effective']}.")
            return self._say(f"Done. {out['addon']} is active and the {out['price']} rupees will be {out['charged']}.")

        if name == "transfer_to_human":
            return self._say("I am connecting you to a human agent now and sharing what we discussed. Please hold.")

        if name == "search_knowledge":
            self.goal = None
            if not out["results"]:
                return self._say("I am not sure about that. Would you like me to transfer you to a human agent?")
            sentences = re.split(r"(?<=[.])\s+", out["results"][0]["text"])
            return self._say(" ".join(sentences[:2]))

        return self._say("Done.")
