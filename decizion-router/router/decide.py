"""
Turning five model answers into one routing decision Codex can act on.

Three things happen here that the model does not do by itself:

1. Consistency. The five questions are answered independently, so they can contradict each other
   (route=tool with reasoning=xhigh). Contradictions are resolved in the safe direction.
2. Policy gates. A decision is only marked auto-appliable when the champion's measured recall for
   that decision passed its gate. Until xhigh recall is high enough, the router still reports
   "xhigh" but tells Codex not to act on it automatically -- advice, not control.
3. Cost asymmetry. Under-routing (hard task -> weak agent) costs correctness; over-routing costs
   tokens. When in doubt the router rounds up, and every round-up is logged so over-escalation is
   measurable too.
"""
import re
from typing import Any, Dict, List, Optional, Tuple

from .risk import authorize_route, classify_tool, risk_flags
from .schema import AGENT_LEVELS, REASONING_LEVELS, Decision, safe_default

def _idx(levels: List[str], value: Optional[str], default: int = 0) -> int:
    return levels.index(value) if value in levels else default


class Policy:
    """Which decisions may be auto-applied, and the floors that protect against under-routing."""

    def __init__(self, cfg: Dict[str, Any], metrics: Dict[str, Any]):
        self.cfg = cfg
        self.metrics = metrics or {}
        gates = cfg.get("auto_apply_gates", {})
        self.gates = gates
        self.applied = {d: self._passes(d, g) for d, g in gates.items()}
        self.min_reasoning_on_risk = cfg.get("min_reasoning_on_risk", "high")
        self.min_agent_on_risk = cfg.get("min_agent_on_risk", "normal_code")
        self.low_confidence = float(cfg.get("low_confidence", 0.5))

    def _passes(self, decision: str, gate: Dict[str, Any]) -> bool:
        for metric, threshold in (gate or {}).items():
            value = self.metrics.get(metric)
            if value is None or float(value) < float(threshold):
                return False
        return True

    def explain(self) -> Dict[str, Any]:
        out = {}
        for d, gate in self.gates.items():
            out[d] = {"auto_apply": self.applied.get(d, False),
                      "gate": gate,
                      "measured": {m: self.metrics.get(m) for m in (gate or {})}}
        return out


def assemble(parsed: Dict[str, Any], latency_ms: float, model_version: str, policy: Policy,
             request_text: str, request_id: Optional[str] = None,
             candidate_tools: Optional[Dict[str, str]] = None) -> Tuple[Decision, Dict[str, Any]]:
    """Build the response. Returns (decision, audit) where audit explains every adjustment made."""
    audit: Dict[str, Any] = {"adjustments": [], "risk_flags": risk_flags(request_text)}

    route = parsed["route"]["value"]
    agent = parsed["agent"]["value"]
    reasoning = parsed["reasoning"]["value"]
    retrieval = bool(parsed["retrieval"]["value"])
    clarify = bool(parsed["clarify"]["value"])
    conf = {k: round(float(v["confidence"]), 4) for k, v in parsed.items() if k in
            ("route", "agent", "reasoning", "retrieval", "clarify")}

    # -- consistency between independently answered questions ---------------
    # route and clarify are asked independently, so they can disagree. Trust the CONFIDENT one:
    # an unsure "clarify" route next to a confident "this is not ambiguous" is the route being wrong.
    if route == "clarify" and not clarify and parsed["clarify"]["confidence"] > conf["route"]:
        probs = parsed["route"].get("probabilities", {})
        alt = max((k for k in probs if k != "clarify"), key=lambda k: probs[k], default="agent")
        audit["adjustments"].append(
            f"route clarify -> {alt} (clarify says not ambiguous at {parsed['clarify']['confidence']:.2f} "
            f"vs route confidence {conf['route']:.2f})")
        route = alt
    elif route == "clarify" and not clarify:
        clarify = True
        audit["adjustments"].append("clarify forced true because route=clarify")
    if clarify and route == "agent" and parsed["clarify"]["confidence"] > 0.9 and conf["route"] < 0.9:
        route = "clarify"
        audit["adjustments"].append("route -> clarify (confident clarify, unsure route)")
    if route == "tool" and _idx(REASONING_LEVELS, reasoning) > 1:
        # a tool call that supposedly needs deep reasoning is a contradiction: trust the harder read
        route = "agent"
        audit["adjustments"].append(f"route tool -> agent (reasoning={reasoning} contradicts a tool call)")
    if route != "agent":
        agent = None
    if route == "agent" and agent is None:
        agent = "normal_code"
        audit["adjustments"].append("agent defaulted to normal_code")

    # -- cost asymmetry: round up, never down -------------------------------
    if audit["risk_flags"] and route == "agent":
        floor_r = _idx(REASONING_LEVELS, policy.min_reasoning_on_risk, 2)
        if _idx(REASONING_LEVELS, reasoning) < floor_r:
            audit["adjustments"].append(
                f"reasoning {reasoning} -> {REASONING_LEVELS[floor_r]} (risk: {','.join(audit['risk_flags'])})")
            reasoning = REASONING_LEVELS[floor_r]
        floor_a = _idx(AGENT_LEVELS, policy.min_agent_on_risk, 1)
        if _idx(AGENT_LEVELS, agent) < floor_a:
            audit["adjustments"].append(
                f"agent {agent} -> {AGENT_LEVELS[floor_a]} (risk: {','.join(audit['risk_flags'])})")
            agent = AGENT_LEVELS[floor_a]
    if route == "agent" and conf["agent"] < policy.low_confidence and _idx(AGENT_LEVELS, agent) < 2:
        up = AGENT_LEVELS[_idx(AGENT_LEVELS, agent) + 1]
        audit["adjustments"].append(f"agent {agent} -> {up} (confidence {conf['agent']:.2f} below floor)")
        agent = up

    # agent and reasoning must not disagree by more than one step
    if route == "agent":
        gap = _idx(AGENT_LEVELS, agent) - _idx(REASONING_LEVELS, reasoning) + 1
        audit["agent_reasoning_disagreement"] = abs(gap) > 1
        if gap > 1:  # deep_code with skip/medium reasoning: raise the reasoning
            target = REASONING_LEVELS[min(3, _idx(AGENT_LEVELS, agent) + 1)]
            audit["adjustments"].append(f"reasoning {reasoning} -> {target} (disagrees with agent={agent})")
            reasoning = target
    else:
        audit["agent_reasoning_disagreement"] = False

    # -- authorization: what may be APPLIED, which is not the same as what was predicted ----
    tool = parsed.get("tool", {}).get("value")
    tool_class = classify_tool(tool, (candidate_tools or {}).get(tool, "")) if tool else "unknown"
    auth = authorize_route(route, tool, tool_class, conf["route"], audit["risk_flags"],
                           policy.cfg, policy.applied.get("route", False))
    applied = dict(policy.applied)
    applied["route"] = auth.route          # risk can only take route authorization away
    audit["authorization"] = auth.as_dict()

    d = Decision(route=route, agent=agent, reasoning=reasoning, retrieval=retrieval, clarify=clarify,
                 tool=tool, confidence=conf, latency_ms=round(latency_ms, 2),
                 model_version=model_version, applied_policy=applied,
                 risk_flags=audit["risk_flags"], tool_class=tool_class if tool else None,
                 requires_human_approval=auth.requires_human_approval,
                 authorization_reasons=auth.reasons, request_id=request_id)
    if "tool" in parsed:
        d.confidence["tool"] = round(float(parsed["tool"]["confidence"]), 4)
    return d, audit


def state_from(req) -> Dict[str, Any]:
    """The state shape the model was trained on."""
    state: Dict[str, Any] = {"request": req.request}
    if req.project_context:
        state["project_context"] = req.project_context
    if req.recent_turns:
        state["recent_turns"] = list(req.recent_turns)
    return state


__all__ = ["Policy", "assemble", "state_from", "risk_flags", "safe_default"]
