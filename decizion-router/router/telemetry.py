"""
Structured decision log, live metrics, and the active-learning triage that feeds the next dataset.

Every decision is written as one JSON line with everything needed to replay it later: the input,
the raw model answers, the adjustments the policy made, and the version that answered. That file is
the raw material for the next dataset, so it is written even in shadow mode.

Triage decides which decisions a human should look at. It does NOT write training data -- it writes
a review queue. Labels come from review, never from the model's own output, or the next version
would just learn this version's mistakes.
"""
import hashlib
import json
import logging
import random
import threading
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("router.telemetry")
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


class DecisionLog:
    """Append-only JSONL, one line per decision."""

    def __init__(self, path: Optional[Path] = None):
        self.path = path or (DATA / "decisions.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")


class ReviewQueue:
    """Candidates for human labelling, deduplicated by request text."""

    def __init__(self, path: Optional[Path] = None, keep_easy_ratio: float = 0.1):
        self.path = path or (DATA / "review_queue.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.keep_easy_ratio = keep_easy_ratio
        self._lock = threading.Lock()
        self._seen = set()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    self._seen.add(json.loads(line)["fingerprint"])
                except Exception:
                    continue

    @staticmethod
    def fingerprint(text: str) -> str:
        norm = " ".join((text or "").lower().split())
        return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]

    def offer(self, record: Dict[str, Any], reasons: List[str],
              outcome: Optional[str] = None) -> bool:
        fp = self.fingerprint(record.get("request", ""))
        with self._lock:
            if fp in self._seen:
                return False
            self._seen.add(fp)
            item = {"fingerprint": fp, "queued_at": time.time(), "reasons": reasons,
                    "priority": priority_of(reasons),
                    "status": "pending", "label": None,
                    "request_id": record.get("request_id"),
                    "request": record.get("request"), "project_context": record.get("project_context"),
                    "recent_turns": record.get("recent_turns"),
                    "model_version": record.get("model_version"),
                    "prediction": record.get("decision"), "confidence": record.get("confidence"),
                    "probabilities": record.get("probabilities"),
                    "risk_flags": (record.get("audit") or {}).get("risk_flags"),
                    "policy_adjustments": (record.get("audit") or {}).get("adjustments"),
                    "outcome": outcome}
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
        return True

    def record_outcome(self, request_id: str, outcome: str) -> int:
        """Attach what actually happened to a queued item, so review sees prediction vs reality."""
        if not self.path.exists():
            return 0
        lines, n = [], 0
        with self._lock:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    it = json.loads(line)
                except Exception:
                    lines.append(line)
                    continue
                if it.get("request_id") == request_id and it.get("outcome") is None:
                    it["outcome"] = outcome
                    n += 1
                lines.append(json.dumps(it, ensure_ascii=False, default=str))
            body = chr(10).join(lines)
            self.path.write_text(body + (chr(10) if lines else ""), encoding="utf-8")
        return n

    def pending(self, priority: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                it = json.loads(line)
            except Exception:
                continue
            if it.get("status") == "pending" and (priority is None or it.get("priority") == priority):
                out.append(it)
        return out

    def counts(self) -> Dict[str, int]:
        if not self.path.exists():
            return {}
        c = Counter()
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                it = json.loads(line)
            except Exception:
                continue
            c[it.get("status", "pending")] += 1
            if it.get("status") == "pending":
                c[it.get("priority", "P2")] += 1
            for r in it.get("reasons", []):
                c["reason:" + r] += 1
        return dict(c)


# Active-learning priorities. P0 is where being wrong is expensive and silent, so those cases are
# reviewed first and are the ones that actually move the next model.
P0, P1, P2 = "P0", "P1", "P2"
PRIORITY_OF = {
    # P0 -- expensive, silent, or unsafe
    "xhigh_underestimate": P0, "deep_code_underestimate": P0, "migration_risk": P0,
    "false_fast_path": P0, "risk_data_loss": P0, "risk_security": P0, "risk_migration": P0,
    "risk_infrastructure": P0, "risk_database": P0, "task_outcome_failed": P0,
    "task_outcome_escalated": P0, "degraded": P0,
    # P1 -- the model is unsure or inconsistent
    "low_confidence": P1, "high_entropy": P1, "agent_reasoning_disagreement": P1,
    "near_threshold": P1, "task_outcome_wrong_agent": P1, "task_outcome_wrong_reasoning": P1,
    "risk_concurrency": P1, "risk_architecture": P1, "policy_adjusted": P1,
    # P2 -- wasteful but safe
    "over_escalation": P2, "unnecessary_retrieval": P2, "routine_sample": P2,
}


def priority_of(reasons: List[str]) -> str:
    """The most severe priority among the reasons."""
    for level in (P0, P1, P2):
        if any(PRIORITY_OF.get(r) == level for r in reasons):
            return level
    return P2


def triage(decision, audit: Dict[str, Any], policy, near: float = 0.15) -> List[str]:
    """Why a human should look at this decision. Empty list = no review needed.

    Under-escalation cannot be detected here without a ground truth, so it is inferred from the
    signals that correlate with it: a risky request routed to a fast path, a confident-looking
    answer that the policy had to correct, a later failure reported through report_outcome.

    A queue made only of hard cases teaches the next model that everything is hard, so a small
    random sample of confident, unremarkable decisions is kept too.
    """
    reasons: List[str] = []
    conf = decision.confidence
    flags = audit.get("risk_flags", [])

    # -- P0: expensive and silent ------------------------------------------
    risky = bool(flags)
    fast_path = decision.route == "tool" or decision.agent == "fast_code"
    if risky and fast_path:
        reasons.append("false_fast_path")
    if "migration" in flags:
        reasons.append("migration_risk")
        if decision.reasoning != "xhigh":
            reasons.append("xhigh_underestimate")
        if decision.route == "agent" and decision.agent != "deep_code":
            reasons.append("deep_code_underestimate")
    if risky and decision.route == "agent" and decision.reasoning in ("skip", "medium"):
        reasons.append("xhigh_underestimate")
    if risky and decision.route == "agent" and decision.agent == "fast_code":
        reasons.append("deep_code_underestimate")
    for flag in flags:
        reasons.append("risk_" + flag)
    if decision.degraded:
        reasons.append("degraded")

    # -- P1: unsure or inconsistent ----------------------------------------
    # Only the decisions that drive behaviour, and only when clearly unsure. Flagging any of the
    # five confidences near 0.5 captured ~90% of traffic here, which is a firehose, not a queue.
    acting = [conf.get(k, 1.0) for k in ("route", "agent") if conf.get(k) is not None]
    if acting and min(acting) < policy.low_confidence:
        reasons.append("low_confidence")
    elif acting and min(acting) < policy.low_confidence + near:
        reasons.append("near_threshold")
    if audit.get("agent_reasoning_disagreement"):
        reasons.append("agent_reasoning_disagreement")
    # an adjustment that moved an agent or reasoning LEVEL is worth a look; the route/clarify
    # cross-check firing is routine
    if any(("->" in a and ("agent" in a or "reasoning" in a)) for a in audit.get("adjustments", [])):
        reasons.append("policy_adjusted")

    # -- P2: wasteful but safe ---------------------------------------------
    if decision.reasoning == "xhigh" and not risky and conf.get("reasoning", 1.0) < 0.8:
        reasons.append("over_escalation")
    if decision.route == "tool" and decision.retrieval is False and decision.reasoning == "skip":
        pass  # the cheap, ordinary case
    if not reasons and random.random() < 0.05:
        reasons.append("routine_sample")
    return list(dict.fromkeys(reasons))      # stable order, no duplicates


class Metrics:
    """In-process counters and latency windows, exposed on /metrics."""

    def __init__(self, window: int = 2000):
        self._lock = threading.Lock()
        self.started = time.time()
        self.latency = deque(maxlen=window)
        self.counts = Counter()
        self.by_route = Counter()
        self.by_agent = Counter()
        self.by_reasoning = Counter()
        self.adjust = Counter()
        self.outcomes = Counter()

    def observe(self, decision, audit: Dict[str, Any], shadow: bool = False) -> None:
        with self._lock:
            self.counts["requests"] += 1
            if shadow:
                self.counts["shadow"] += 1
            if decision.degraded:
                self.counts["degraded"] += 1
            self.latency.append(decision.latency_ms)
            self.by_route[decision.route] += 1
            if decision.agent:
                self.by_agent[decision.agent] += 1
            self.by_reasoning[decision.reasoning] += 1
            if decision.route == "agent" and decision.agent == "fast_code" and audit.get("risk_flags"):
                self.counts["possible_false_fast_path"] += 1
            if audit.get("agent_reasoning_disagreement"):
                self.counts["agent_reasoning_disagreement"] += 1
            for a in audit.get("adjustments", []):
                self.adjust[a.split(" (")[0]] += 1

    def count_outcome(self, outcome: str) -> None:
        with self._lock:
            self.outcomes[outcome] += 1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            lat = sorted(self.latency)
            q = lambda p: round(lat[min(len(lat) - 1, int(len(lat) * p))], 2) if lat else None
            n = max(1, self.counts["requests"])
            return {
                "uptime_s": round(time.time() - self.started, 1),
                "requests": self.counts["requests"],
                "shadow_requests": self.counts["shadow"],
                "degraded": self.counts["degraded"],
                "degraded_rate": round(self.counts["degraded"] / n, 4),
                "latency_ms": {"p50": q(0.5), "p95": q(0.95), "p99": q(0.99)},
                "route": dict(self.by_route),
                "agent": dict(self.by_agent),
                "reasoning": dict(self.by_reasoning),
                "possible_false_fast_path_rate": round(self.counts["possible_false_fast_path"] / n, 4),
                "agent_reasoning_disagreement_rate": round(self.counts["agent_reasoning_disagreement"] / n, 4),
                "policy_adjustments": dict(self.adjust),
                "reported_outcomes": dict(self.outcomes),
            }
