"""
The running router: config, the resident models (champion and challenger), shadow mode and canary.

Shadow mode and canary are different things and both are here:
  * shadow  -- the challenger answers every request too, its answer is logged and compared, and
               nobody acts on it. Zero risk, produces the data that justifies a promotion.
  * canary  -- a percentage of real traffic is actually served by the challenger. Risk, but it is
               the only way to see the challenger under production conditions.
A request is assigned to the canary by hashing its id, so the same request always gets the same
model and a client retrying does not flip between versions.
"""
import hashlib
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from . import registry
from .decide import Policy, assemble, state_from
from .model import RouterModel, build
from .schema import Decision, RouteRequest, safe_default
from .telemetry import DecisionLog, Metrics, ReviewQueue, triage

logger = logging.getLogger("router.service")
ROOT = Path(__file__).resolve().parent.parent
SIMULATION_PREFIX = "sim-"

DEFAULT_CONFIG = {
    "device": "cuda",
    "shadow_mode": True,
    "low_confidence": 0.5,
    "min_reasoning_on_risk": "high",
    "min_agent_on_risk": "normal_code",
    # A decision is auto-appliable only when the champion measured at least this on the eval suite.
    # xhigh routing stays advisory until its recall is high enough to trust unattended.
    "auto_apply_gates": {
        "route": {"route_accuracy": 0.85},
        "agent": {"agent_accuracy": 0.80, "deep_code_recall": 0.80},
        "reasoning": {"reasoning_accuracy": 0.80, "xhigh_recall": 0.90},
        "retrieval": {"retrieval_accuracy": 0.80},
        "clarify": {"clarify_accuracy": 0.85},
    },
}


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    p = Path(path or os.getenv("ROUTER_CONFIG") or (ROOT / "config" / "config.yaml"))
    if p.exists():
        loaded = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        gates = {**cfg["auto_apply_gates"], **(loaded.pop("auto_apply_gates", {}) or {})}
        cfg.update(loaded)
        cfg["auto_apply_gates"] = gates
        cfg["_config_file"] = str(p)
    return cfg


class RouterService:
    def __init__(self, config_path: Optional[str] = None):
        self.cfg = load_config(config_path)
        self.log = DecisionLog()
        self.queue = ReviewQueue()
        self.metrics = Metrics()
        self.champion: Optional[RouterModel] = None
        self.challenger: Optional[RouterModel] = None
        self.grep_model: Optional[RouterModel] = None
        self.policy: Optional[Policy] = None
        self.started = time.time()
        self._load_lock = threading.Lock()
        self.last_error: Optional[str] = None

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> None:
        with self._load_lock:
            mv = registry.champion()
            if mv is None:
                raise RuntimeError("no champion registered -- run scripts/bootstrap.py first")
            self.champion = build(mv, device=self.cfg["device"])
            self.champion.load()
            self.policy = Policy(self.cfg, mv.metrics)
            grep_checkpoint = ROOT.parent / "finetune" / "ckpt_decision_grep_v1.pt"
            if grep_checkpoint.exists():
                try:
                    self.grep_model = RouterModel(
                        checkpoint=str(grep_checkpoint), base_model="multilingual", dtype="fp16",
                        max_len=1024, questions_version=1, version="decision-grep-v1",
                        device=self.cfg["device"])
                    self.grep_model.load()
                except Exception as e:
                    logger.error("decision-grep-v1 failed to load; typed requests can still use the champion: %s", e)
                    self.grep_model = None
            ch = registry.challenger()
            if ch is not None and (self.cfg.get("shadow_mode") or registry.canary_percent() > 0):
                try:
                    self.challenger = build(ch, device=self.cfg["device"])
                    self.challenger.load()
                except Exception as e:      # a broken challenger must never stop the champion
                    logger.error("challenger %s failed to load: %s", ch.version, e)
                    self.challenger = None

    @property
    def ready(self) -> bool:
        return self.champion is not None and self.champion.ready

    def reload(self) -> Dict[str, Any]:
        """Pick up a registry change (promotion, rollback, new challenger) without a restart."""
        self.champion = self.challenger = None
        self.start()
        return self.status()

    # -- serving ------------------------------------------------------------
    @staticmethod
    def _in_canary(request_id: Optional[str], percent: int) -> bool:
        if percent <= 0:
            return False
        if percent >= 100:
            return True
        key = (request_id or str(time.time())).encode("utf-8")
        return int(hashlib.sha1(key).hexdigest()[:8], 16) % 100 < percent

    def _run(self, model: RouterModel, req: RouteRequest):
        state = state_from(req)
        parsed, ms = model.decide_raw(state, req.candidate_tools)
        d, audit = assemble(parsed, ms, model.version, self.policy, req.request, req.request_id,
                            req.candidate_tools)
        # full distributions, not just the winner: a promotion argument needs them, and so does
        # anyone later asking how close the call was
        probs = {k: (v.get("probabilities") or ({"p_true": v["p_true"]} if "p_true" in v else {}))
                 for k, v in parsed.items()}
        audit["state_tokens"] = model.count_tokens(state)
        return d, audit, probs

    def route(self, req: RouteRequest) -> Decision:
        if not self.ready:
            d = safe_default(req.request, "router not ready")
            self.metrics.observe(d, {}, shadow=False)
            return d

        canary_pct = registry.canary_percent()
        use_challenger = self.challenger is not None and self._in_canary(req.request_id, canary_pct)
        serving = self.challenger if use_challenger else self.champion

        t0 = time.perf_counter()
        try:
            decision, audit, probs = self._run(serving, req)
        except Exception as e:                       # never let the router take Codex down
            self.last_error = f"{type(e).__name__}: {e}"
            logger.exception("inference failed")
            decision = safe_default(req.request, self.last_error, (time.perf_counter() - t0) * 1000,
                                    serving.version if serving else "unknown")
            audit, probs = {"risk_flags": [], "adjustments": []}, {}

        audit["served_by"] = serving.version if serving else "fallback"
        audit["canary"] = use_challenger

        # shadow: the challenger answers too, nobody acts on it
        shadow_record = None
        if self.challenger is not None and not use_challenger and self.cfg.get("shadow_mode"):
            try:
                sd, _sa, _sp = self._run(self.challenger, req)
                shadow_record = {"version": sd.model_version, "route": sd.route, "agent": sd.agent,
                                 "reasoning": sd.reasoning, "retrieval": sd.retrieval,
                                 "clarify": sd.clarify, "confidence": sd.confidence,
                                 "latency_ms": sd.latency_ms,
                                 "agrees": (sd.route, sd.agent, sd.reasoning) ==
                                           (decision.route, decision.agent, decision.reasoning)}
            except Exception as e:
                shadow_record = {"error": str(e)}

        # simulation/demo traffic (e.g. the traffic game) must never reach the decision log or the
        # review queue -- that data trains the next model version
        if (req.request_id or "").startswith(SIMULATION_PREFIX):
            return decision

        self.metrics.observe(decision, audit, shadow=False)
        record = {
            "ts": time.time(), "request_id": req.request_id,
            "request": req.request, "project_context": req.project_context,
            "recent_turns": req.recent_turns,
            "candidate_tools": list(req.candidate_tools) if req.candidate_tools else None,
            "model_version": decision.model_version, "served_by": audit["served_by"],
            "canary": use_challenger,
            "decision": {"route": decision.route, "agent": decision.agent,
                         "reasoning": decision.reasoning, "retrieval": decision.retrieval,
                         "clarify": decision.clarify, "tool": decision.tool},
            "confidence": decision.confidence, "probabilities": probs,
            "state_tokens": audit.get("state_tokens"),
            "risk_flags": decision.risk_flags, "tool_class": decision.tool_class,
            "requires_human_approval": decision.requires_human_approval,
            "applied_policy": decision.applied_policy,
            "latency_ms": decision.latency_ms,
            "degraded": decision.degraded, "audit": audit, "shadow": shadow_record,
        }
        reasons = triage(decision, audit, self.policy)
        if reasons:
            record["review_reasons"] = reasons
            self.queue.offer(record, reasons)
        self.log.write(record)
        return decision

    def system_one(self, state: Any, questions: Dict[str, Dict[str, Any]],
                   model_name: Optional[str] = None) -> Dict[str, Any]:
        """Serve arbitrary typed decisions without changing the fixed Codex routing contract."""
        if not self.ready or self.champion is None:
            raise RuntimeError("router not ready")
        grep_request = (
            model_name == "decision-grep-v1"
            and isinstance(state, dict)
            and isinstance(state.get("items"), list)
            and bool(questions)
            and all(re.fullmatch(r"q\d+", key) for key in questions)
        )
        if grep_request:
            if self.grep_model is None or not self.grep_model.ready:
                raise RuntimeError("decision-grep-v1 checkpoint is missing or failed to load")
            selected = self.grep_model
            # Train and infer on one candidate at a time. The input index is the
            # qN -> nN pairing used by Jevgrep; remap it to n0 for the isolated example.
            answers, usage, latency_ms = {}, {}, 0.0
            for key, question in questions.items():
                index = int(key[1:])
                if index >= len(state["items"]) or not isinstance(state["items"][index], dict):
                    raise ValueError(f"no candidate item for question {key}")
                candidate = {**state["items"][index], "id": "n0"}
                isolated_state = {**state, "items": [candidate]}
                isolated_question = {
                    **question,
                    "instructions": str(question.get("instructions", "")) + " Candidate id: n0.",
                }
                result, elapsed = selected.system_one(isolated_state, {key: isolated_question})
                answers.update(result.get("answers", {}))
                latency_ms += elapsed
                for name, value in (result.get("usage") or {}).items():
                    if isinstance(value, (int, float)):
                        usage[name] = usage.get(name, 0) + value
            result = {"answers": answers}
            if usage:
                result["usage"] = usage
        else:
            selected = self.champion
            result, latency_ms = selected.system_one(state, questions)
        result["model"] = selected.version
        result["latency_ms"] = round(latency_ms, 2)
        return result

    def feedback(self, request_id: str, outcome: str, note: str = "",
                 actual_agent: Optional[str] = None, actual_reasoning: Optional[str] = None,
                 needed_more_retrieval: Optional[bool] = None) -> bool:
        """Report what actually happened (task failed, wrong agent, had to escalate...).

        A failure after the fact is the strongest signal there is, so it always gets reviewed.
        """
        rec = {"ts": time.time(), "request_id": request_id, "outcome": outcome, "note": note,
               "kind": "feedback", "actual_agent": actual_agent,
               "actual_reasoning": actual_reasoning, "needed_more_retrieval": needed_more_retrieval}
        self.log.write(rec)
        # attach the outcome to the queued prediction, so review sees predicted vs actual together
        linked = self.queue.record_outcome(request_id, outcome)
        if outcome in ("failed", "escalated", "wrong_agent", "wrong_reasoning"):
            self.metrics.count_outcome(outcome)
            if not linked:
                # confident enough not to be queued, and still wrong: exactly what the next
                # dataset needs most
                o = self._find_decision(request_id) or {}
                self.queue.offer({"request": o.get("request") or note or request_id,
                                  "request_id": request_id,
                                  "project_context": o.get("project_context"),
                                  "recent_turns": o.get("recent_turns"),
                                  "model_version": o.get("model_version", "unknown"),
                                  "decision": o.get("decision"), "confidence": o.get("confidence"),
                                  "probabilities": o.get("probabilities"), "audit": o.get("audit")},
                                 ["task_outcome_" + outcome], outcome=outcome)
        return True

    def _find_decision(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Last logged decision for this request id."""
        try:
            found = None
            with self.log.path.open(encoding="utf-8") as f:
                for line in f:
                    if request_id not in line:
                        continue
                    rec = json.loads(line)
                    if rec.get("request_id") == request_id and rec.get("decision"):
                        found = rec
            return found
        except Exception:
            return None

    # -- introspection ------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        reg = registry.status()
        return {
            "ready": self.ready,
            "uptime_s": round(time.time() - self.started, 1),
            "champion": self.champion.version if self.champion else None,
            "decision_grep": self.grep_model.version if self.grep_model and self.grep_model.ready else None,
            "challenger": self.challenger.version if self.challenger else None,
            "canary_percent": reg["canary_percent"],
            "shadow_mode": bool(self.cfg.get("shadow_mode")),
            "device": self.champion.device if self.champion else self.cfg["device"],
            "dtype": self.cfg.get("dtype", "fp16"),
            "vram_mb": self.champion.vram_mb() if self.champion else None,
            "max_len": self.champion.max_len if self.champion else None,
            "load_seconds": round(self.champion.load_seconds, 1) if self.champion and self.champion.load_seconds else None,
            "auto_apply": self.policy.applied if self.policy else {},
            "registry": reg,
            "last_error": self.last_error,
            "config_file": self.cfg.get("_config_file"),
        }
