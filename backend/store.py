"""Mock telecom backend.

Stands in for real billing / CRM / network systems. All data is fake.
Swap these functions for HTTP calls to real systems later; the agent's tools
only depend on the function signatures here.
"""
from __future__ import annotations

import copy
import itertools

PLANS = {
    "PRE_299": {"name": "Prepaid 299", "type": "prepaid", "price": 299, "data": "1.5 GB per day",
                "validity": "28 days", "data_total_gb": 42.0},
    "PRE_719": {"name": "Prepaid 719", "type": "prepaid", "price": 719, "data": "1.5 GB per day",
                "validity": "84 days", "data_total_gb": 126.0},
    "POST_399": {"name": "Postpaid 399", "type": "postpaid", "price": 399, "data": "40 GB per month",
                 "validity": "monthly", "data_total_gb": 40.0},
    "POST_599": {"name": "Postpaid 599", "type": "postpaid", "price": 599,
                 "data": "75 GB per month with unlimited 5G", "validity": "monthly", "data_total_gb": 75.0},
    "POST_999": {"name": "Postpaid 999 Family", "type": "postpaid", "price": 999,
                 "data": "200 GB shared, 3 connections", "validity": "monthly", "data_total_gb": 200.0},
}

ADDONS = {
    "DATA_1GB": {"name": "1 GB Data Pack", "price": 19, "kind": "data", "extra_gb": 1.0, "validity": "1 day"},
    "DATA_2GB": {"name": "2 GB Data Pack", "price": 29, "kind": "data", "extra_gb": 2.0, "validity": "2 days"},
    "DATA_5GB": {"name": "5 GB Data Pack", "price": 98, "kind": "data", "extra_gb": 5.0, "validity": "7 days"},
    "ROAM_INTL_7D": {"name": "International Roaming 7 Days", "price": 599, "kind": "roaming",
                     "extra_gb": 0.0, "validity": "7 days"},
    "ROAM_INTL_30D": {"name": "International Roaming 30 Days", "price": 1799, "kind": "roaming",
                      "extra_gb": 0.0, "validity": "30 days"},
}

# Towers with a known problem, keyed by pincode.
OUTAGES = {
    "560103": "Planned tower maintenance in your area. Service should be restored by 6 PM today.",
}

_INITIAL_CUSTOMERS = {
    "9876543210": {
        "name": "Aarav Sharma", "pin": "1234", "type": "postpaid", "plan_id": "POST_599",
        "pincode": "560103", "data_used_gb": 61.2, "extra_gb": 0.0, "wallet": None,
        "cycle_end": "2026-10-14", "addons": [],
        "bill": {
            "period": "September 2026", "due_date": "2026-10-05", "total": 1413.64, "previous_total": 706.82,
            "lines": [
                {"item": "Postpaid 599 plan", "amount": 599.00},
                {"item": "International Roaming 7 Days (used 12 to 19 Sep)", "amount": 599.00},
                {"item": "GST at 18 percent", "amount": 215.64},
            ],
        },
    },
    "9123456780": {
        "name": "Priya Nair", "pin": "4321", "type": "prepaid", "plan_id": "PRE_299",
        "pincode": "560001", "data_used_gb": 12.5, "extra_gb": 0.0, "wallet": 45.0,
        "cycle_end": "2026-10-09", "addons": [],
        "bill": None,
    },
    "9988776655": {
        "name": "Rohan Mehta", "pin": "2468", "type": "postpaid", "plan_id": "POST_399",
        "pincode": "560034", "data_used_gb": 33.9, "extra_gb": 0.0, "wallet": None,
        "cycle_end": "2026-10-20", "addons": [],
        "bill": {
            "period": "September 2026", "due_date": "2026-10-10", "total": 470.82, "previous_total": 470.82,
            "lines": [
                {"item": "Postpaid 399 plan", "amount": 399.00},
                {"item": "GST at 18 percent", "amount": 71.82},
            ],
        },
    },
}

CUSTOMERS: dict = {}
TICKETS: list = []
_ticket_ids = itertools.count(1001)


def reset() -> None:
    """Restore the initial data (used by evals and the /api/reset endpoint)."""
    global _ticket_ids
    CUSTOMERS.clear()
    CUSTOMERS.update(copy.deepcopy(_INITIAL_CUSTOMERS))
    TICKETS.clear()
    _ticket_ids = itertools.count(1001)


reset()


# ---------- read operations ----------

def customer_exists(phone: str) -> bool:
    return phone in CUSTOMERS


def check_pin(phone: str, pin: str) -> bool:
    c = CUSTOMERS.get(phone)
    return bool(c) and c["pin"] == pin


def get_account(phone: str) -> dict:
    c = CUSTOMERS[phone]
    plan = PLANS[c["plan_id"]]
    return {"name": c["name"], "type": c["type"], "plan_id": c["plan_id"], "plan_name": plan["name"],
            "plan_price": plan["price"], "active_addons": [ADDONS[a]["name"] for a in c["addons"]]}


def get_usage(phone: str) -> dict:
    c = CUSTOMERS[phone]
    plan = PLANS[c["plan_id"]]
    total = plan["data_total_gb"] + c["extra_gb"]
    out = {
        "plan_name": plan["name"], "data_used_gb": round(c["data_used_gb"], 1),
        "data_total_gb": round(total, 1),
        "data_left_gb": round(max(total - c["data_used_gb"], 0), 1),
        "cycle_end": c["cycle_end"],
    }
    if c["wallet"] is not None:
        out["wallet_balance_rupees"] = c["wallet"]
    return out


def get_bill(phone: str) -> dict:
    c = CUSTOMERS[phone]
    if c["bill"] is None:
        return {"type": "prepaid", "note": "Prepaid account, there is no monthly bill.",
                "wallet_balance_rupees": c["wallet"]}
    b = copy.deepcopy(c["bill"])
    b["type"] = "postpaid"
    b["difference_from_previous"] = round(b["total"] - b["previous_total"], 2)
    return b


def list_plans(customer_type: str | None = None) -> list[dict]:
    return [{"plan_id": pid, **p} for pid, p in PLANS.items()
            if customer_type is None or p["type"] == customer_type]


def list_addons() -> list[dict]:
    return [{"addon_id": aid, **a} for aid, a in ADDONS.items()]


def network_status(phone: str) -> dict:
    c = CUSTOMERS[phone]
    outage = OUTAGES.get(c["pincode"])
    if outage:
        return {"outage": True, "message": outage, "signal_quality": "poor"}
    return {"outage": False, "message": "No outage found in your area.", "signal_quality": "good",
            "suggested_steps": ["Turn airplane mode on for 10 seconds, then off",
                                "Restart the phone", "Check that mobile data is switched on"]}


# ---------- write operations ----------

def change_plan(phone: str, plan_id: str) -> dict:
    c = CUSTOMERS[phone]
    plan = PLANS.get(plan_id)
    if not plan:
        return {"error": f"Unknown plan {plan_id}"}
    if plan["type"] != c["type"]:
        return {"error": f"Cannot move a {c['type']} customer to a {plan['type']} plan."}
    if plan_id == c["plan_id"]:
        return {"error": "The customer is already on this plan."}
    c["plan_id"] = plan_id
    effective = "immediately" if c["type"] == "prepaid" else "from the next bill cycle"
    return {"status": "changed", "new_plan": plan["name"], "effective": effective}


def activate_addon(phone: str, addon_id: str) -> dict:
    c = CUSTOMERS[phone]
    addon = ADDONS.get(addon_id)
    if not addon:
        return {"error": f"Unknown add-on {addon_id}"}
    if c["type"] == "prepaid":
        if c["wallet"] < addon["price"]:
            return {"error": "Insufficient wallet balance.", "wallet_balance_rupees": c["wallet"],
                    "price": addon["price"]}
        c["wallet"] = round(c["wallet"] - addon["price"], 2)
        charged = "deducted from wallet"
    else:
        charged = "added to the next bill"
    c["addons"].append(addon_id)
    c["extra_gb"] += addon["extra_gb"]
    return {"status": "activated", "addon": addon["name"], "price": addon["price"],
            "validity": addon["validity"], "charged": charged}


def create_ticket(phone: str, category: str, description: str) -> dict:
    tid = f"TKT-{next(_ticket_ids)}"
    TICKETS.append({"id": tid, "phone": phone, "category": category, "description": description})
    return {"ticket_id": tid, "status": "open", "eta": "within 24 hours"}
