"""Scenario evals. Run from the project root:

    python -m evals.run_evals              # uses Claude if ANTHROPIC_API_KEY is set, else mock
    LLM_PROVIDER=mock python -m evals.run_evals

Each scenario scripts the caller's turns, then checks (a) which tools were called and
(b) the final state of the mock backend. Checking state, not wording, keeps the
evals stable across models. Add more scenarios as you find failures (aim for 30 to 50).
"""
from __future__ import annotations

import sys
import time

from agent.agent import VoiceAgent
from agent.tools import Session, run_tool
from backend import store

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

A, B, C = "9876543210 pin 1234", "9123456780 pin 4321", "9988776655 pin 2468"

SCENARIOS = [
    dict(name="usage after verification",
         turns=["How much data have I used this month?", f"My number and pin: {A}"],
         expect={"verify_customer", "get_usage"}),
    dict(name="bill explanation (roaming charge)",
         turns=["Why is my bill so high this month?", A],
         expect={"get_bill_summary"}, reply_has="roaming"),
    dict(name="activate data pack with confirmation",
         turns=["I want to activate a 5GB data pack", C, "yes"],
         expect={"activate_addon"}, check=lambda a: "DATA_5GB" in store.CUSTOMERS["9988776655"]["addons"]),
    dict(name="caller declines, nothing changes",
         turns=["Please add a 2GB data pack", C, "no"],
         check=lambda a: store.CUSTOMERS["9988776655"]["addons"] == []),
    dict(name="international roaming activation",
         turns=["I am travelling, activate international roaming for 7 days", A, "yes"],
         check=lambda a: "ROAM_INTL_7D" in store.CUSTOMERS["9876543210"]["addons"]),
    dict(name="plan change",
         turns=["I want to change my plan", C, "the 599 one", "yes"],
         check=lambda a: store.CUSTOMERS["9988776655"]["plan_id"] == "POST_599"),
    dict(name="network complaint raises ticket",
         turns=["My internet is not working", B],
         expect={"run_network_diagnostic", "create_ticket"}, check=lambda a: len(store.TICKETS) == 1),
    dict(name="known outage, no ticket needed",
         turns=["I have no signal on my phone", A],
         expect={"run_network_diagnostic"}, forbid={"create_ticket"}),
    dict(name="wrong PIN three times locks the session",
         turns=["Check my usage", "9876543210 0000", "9876543210 1111", "9876543210 2222", "9876543210 1234"],
         forbid={"get_usage"}, check=lambda a: a.session.locked),
    dict(name="asks for a human, transfers with summary",
         turns=["I want to talk to a human agent"],
         expect={"transfer_to_human"}, check=lambda a: a.session.transferred and a.session.transfer_info["summary"]),
    dict(name="policy question uses knowledge base",
         turns=["What is the late fee if I miss the due date?"],
         expect={"search_knowledge"}, forbid={"verify_customer"}, reply_has="late"),
    dict(name="out of scope request is declined",
         turns=["What is the weather like today?"],
         forbid_all_tools=True, reply_has="telecom"),
]


def guardrail_checks() -> list[tuple[str, bool]]:
    """Direct tests of the code-level guardrails, no LLM involved."""
    store.reset()
    s = Session()
    results = []
    results.append(("account tool blocked before verification", "error" in run_tool(s, "get_usage", {})))
    run_tool(s, "verify_customer", {"phone_number": "9876543210", "pin": "1234"})
    other = run_tool(s, "get_usage", {"phone_number": "9123456780"})
    results.append(("tools ignore a phone number the model passes",
                    other.get("plan_name") == "Postpaid 599"))
    s.turn = 1
    gate = run_tool(s, "activate_addon", {"addon_id": "DATA_1GB", "confirmed": False})
    results.append(("first call only asks for confirmation", gate.get("status") == "confirmation_required"))
    same_turn = run_tool(s, "activate_addon", {"addon_id": "DATA_1GB", "confirmed": True})
    results.append(("cannot confirm in the same turn", "error" in same_turn))
    s.turn = 2
    done = run_tool(s, "activate_addon", {"addon_id": "DATA_1GB", "confirmed": True})
    results.append(("confirmed in a later turn works", done.get("status") == "activated"))
    replay = run_tool(s, "activate_addon", {"addon_id": "DATA_1GB", "confirmed": True})
    results.append(("confirmation cannot be replayed", "error" in replay))
    s.turn = 3
    run_tool(s, "activate_addon", {"addon_id": "DATA_2GB", "confirmed": False})
    s.turn = 4
    swapped = run_tool(s, "activate_addon", {"addon_id": "DATA_5GB", "confirmed": True})
    results.append(("confirmation for a different item is rejected", "error" in swapped))
    return results


def run_scenario(sc: dict) -> tuple[bool, str, float]:
    store.reset()
    agent = VoiceAgent()
    used: set[str] = set()
    last_reply, slowest = "", 0.0
    for turn in sc["turns"]:
        r = agent.respond(turn)
        used |= {step["tool"] for step in r["trace"]}
        last_reply, slowest = r["reply"], max(slowest, r["latency_ms"])
    problems = []
    if not sc.get("expect", set()) <= used:
        problems.append(f"missing tools {sc['expect'] - used}")
    if sc.get("forbid", set()) & used:
        problems.append(f"forbidden tools used {sc['forbid'] & used}")
    if sc.get("forbid_all_tools") and used:
        problems.append(f"no tools expected, got {used}")
    if sc.get("reply_has") and sc["reply_has"] not in last_reply.lower():
        problems.append(f"reply lacks '{sc['reply_has']}': {last_reply!r}")
    if sc.get("check") and not sc["check"](agent):
        problems.append("state check failed")
    return (not problems, "; ".join(problems), slowest)


def main() -> int:
    print("Guardrails (code level)")
    failed = 0
    guardrails = guardrail_checks()
    for name, ok in guardrails:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        failed += not ok

    agent_probe = VoiceAgent()
    print(f"\nScenarios  [LLM: {agent_probe.llm.name}]")
    t0 = time.time()
    for sc in SCENARIOS:
        ok, why, ms = run_scenario(sc)
        print(f"  {'PASS' if ok else 'FAIL'}  {sc['name']:<48} slowest turn {ms:>5.0f} ms  {why}")
        failed += not ok
    total = len(SCENARIOS) + len(guardrails)
    print(f"\n{total - failed}/{total} passed in {time.time() - t0:.1f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
