"""
MCP server exposing the router to Codex Desktop.

It is a thin client of the internal HTTP API, on purpose: the model stays resident in one process
that Codex does not own, so restarting Codex does not reload 322M parameters, and the same router
can serve other clients.

Every tool fails soft. If the router is down, slow or degraded, the tool still returns a valid
decision with `degraded: true` and `auto_apply` all false, so Codex keeps working with its own
defaults. The router must never be a single point of failure.
"""
import json
import os
import time
from typing import Any, Dict, List, Optional

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Decizion Router")

BASE_URL = os.getenv("ROUTER_URL", "http://127.0.0.1:8099").rstrip("/")
TIMEOUT = float(os.getenv("ROUTER_TIMEOUT", "3.0"))   # a router slower than this is worse than none

FALLBACK = {
    "schema_version": 1, "route": "agent", "agent": "normal_code", "reasoning": "high",
    "retrieval": True, "clarify": False,
    "confidence": {k: 0.0 for k in ("route", "agent", "reasoning", "retrieval", "clarify")},
    "latency_ms": 0.0, "model_version": "fallback", "degraded": True,
    "applied_policy": {k: False for k in ("route", "agent", "reasoning", "retrieval", "clarify")},
}


def _fallback(reason: str) -> Dict[str, Any]:
    d = dict(FALLBACK)
    d["reason"] = reason
    return d


@mcp.tool()
def route_task(request: str, project_context: str = "", recent_turns: Optional[List[str]] = None,
               candidate_tools: Optional[Dict[str, str]] = None, request_id: str = "") -> str:
    """Decide how to handle a user request: tool vs agent vs clarify, which agent, how much reasoning.

    Returns JSON with route, agent, reasoning, retrieval, clarify, per-decision confidence, and
    `applied_policy`: for each decision, whether it is safe to apply automatically. A decision with
    applied_policy=false is ADVICE ONLY -- keep the current default and do not change behaviour.

    `candidate_tools` should be the top-K from a retriever ({tool_id: short description}), not the
    whole catalogue: accuracy drops as the option list grows.
    """
    payload = {"request": request, "request_id": request_id or f"mcp-{time.time_ns()}"}
    if project_context:
        payload["project_context"] = project_context
    if recent_turns:
        payload["recent_turns"] = recent_turns
    if candidate_tools:
        payload["candidate_tools"] = candidate_tools
    try:
        r = httpx.post(f"{BASE_URL}/v1/route", json=payload, timeout=TIMEOUT)
        r.raise_for_status()
        d = r.json()
        if d.get("schema_version") != 1 or "route" not in d:
            return json.dumps(_fallback("unexpected schema from router"), ensure_ascii=False)
        return json.dumps(d, ensure_ascii=False)
    except Exception as e:
        return json.dumps(_fallback(f"{type(e).__name__}: {e}"), ensure_ascii=False)


@mcp.tool()
def report_outcome(request_id: str, outcome: str, note: str = "") -> str:
    """Tell the router what actually happened, so bad routings become training data.

    outcome: ok | failed | escalated | wrong_agent | wrong_reasoning
    Use `escalated` when a task routed to fast/normal had to be redone by a stronger agent -- that is
    the single most valuable signal the router can get.
    """
    try:
        r = httpx.post(f"{BASE_URL}/v1/feedback",
                       json={"request_id": request_id, "outcome": outcome, "note": note},
                       timeout=TIMEOUT)
        r.raise_for_status()
        return json.dumps({"ok": True}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)


@mcp.tool()
def router_status() -> str:
    """Which model is serving, what it is allowed to decide automatically, and how it is performing."""
    try:
        s = httpx.get(f"{BASE_URL}/v1/status", timeout=TIMEOUT).json()
        m = httpx.get(f"{BASE_URL}/metrics", timeout=TIMEOUT).json()
        return json.dumps({"status": s, "metrics": m}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e), "hint": "start it with ./run-router"},
                          ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()
