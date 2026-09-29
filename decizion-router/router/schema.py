"""
The wire contract between the router and its callers (Codex via MCP, or the internal API).

SCHEMA_VERSION is part of every response. Callers must branch on it rather than on the presence
of fields, so the router can add decisions later without breaking a running Codex.
"""
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, model_validator

SCHEMA_VERSION = 1

Route = Literal["tool", "agent", "clarify"]
Agent = Literal["fast_code", "normal_code", "deep_code"]
Reasoning = Literal["skip", "medium", "high", "xhigh"]
REASONING_LEVELS = ["skip", "medium", "high", "xhigh"]
AGENT_LEVELS = ["fast_code", "normal_code", "deep_code"]


class RouteRequest(BaseModel):
    request: str = Field(..., description="The user's message, verbatim.")
    project_context: Optional[str] = Field(None, description="One or two lines about the repo/stack.")
    recent_turns: Optional[list] = Field(None, description="Recent conversation/tool lines, oldest first.")
    candidate_tools: Optional[Dict[str, str]] = Field(
        None, description="Top-K tools from the retriever, {tool_id: short description}.")
    request_id: Optional[str] = None


class SystemOneRequest(BaseModel):
    """Generic typed-decision request, compatible with Jev's SystemOne payload shape."""

    state: Any = Field(..., description="Observed state to evaluate (text, object, or conversation list).")
    questions: Dict[str, Dict[str, Any]] = Field(..., description="Named typed questions evaluated in one pass.")
    model: Optional[str] = Field(None, description="Optional local model selector; decision-grep-v1 selects the code-relevance model.")

    @classmethod
    def validate_questions(cls, questions: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        if not questions or len(questions) > 24:
            raise ValueError("questions must contain between 1 and 24 entries")
        total_options = 0
        for name, question in questions.items():
            if not name or len(name) > 64:
                raise ValueError("question names must contain 1 to 64 characters")
            if not isinstance(question, dict):
                raise ValueError(f"question {name!r} must be an object")
            kind = question.get("type")
            instructions = question.get("instructions")
            if kind not in {"choice", "score", "noul"}:
                raise ValueError(f"question {name!r} has unsupported type")
            if not isinstance(instructions, (str, dict, list)):
                raise ValueError(f"question {name!r} needs instructions")
            criteria = question.get("criteria")
            if kind == "choice":
                count = len(criteria) if isinstance(criteria, (dict, list)) else 0
                if count < 1:
                    raise ValueError(f"choice question {name!r} needs at least one option")
                if isinstance(criteria, list) and any(not isinstance(option, str) for option in criteria):
                    raise ValueError(f"choice question {name!r} list options must be strings")
            elif kind == "score":
                count = len(criteria) if isinstance(criteria, list) else 0
                if count < 2:
                    raise ValueError(f"score question {name!r} needs at least two criteria")
            else:
                if criteria is not None and not isinstance(criteria, dict):
                    raise ValueError(f"noul question {name!r} criteria must be an object")
                count = 2
            total_options += count
            if count > 192:
                raise ValueError(f"question {name!r} exceeds the model's 192-option limit")
        if total_options > 512:
            raise ValueError("questions exceed the 512-option request limit")
        return questions

    @model_validator(mode="after")
    def validate_question_shapes(self):
        self.validate_questions(self.questions)
        return self


class Decision(BaseModel):
    schema_version: int = SCHEMA_VERSION
    route: Route
    agent: Optional[Agent] = Field(None, description="Only set when route == 'agent'.")
    reasoning: Reasoning
    retrieval: bool
    clarify: bool
    tool: Optional[str] = Field(None, description="Chosen tool when candidate_tools was supplied.")
    confidence: Dict[str, float]
    latency_ms: float
    model_version: str
    degraded: bool = Field(False, description="True when the answer came from the fallback, not the model.")
    reason: Optional[str] = Field(None, description="Why it is degraded, when it is.")
    applied_policy: Dict[str, bool] = Field(
        default_factory=dict,
        description="Which decisions the caller may auto-apply. A decision whose recall has not passed "
                    "its gate is returned as advisory only (false), so Codex keeps its own default. "
                    "`route` is additionally withdrawn for risky or irreversible work.")
    risk_flags: list = Field(default_factory=list,
                             description="migration / data_loss / concurrency / security / infrastructure / "
                                         "database / architecture detected in the request.")
    tool_class: Optional[str] = Field(None, description="read_only | write | destructive | unknown.")
    requires_human_approval: bool = Field(
        False, description="True when the action must go through Codex's own approval path. Model "
                           "confidence never overrides this.")
    authorization_reasons: list = Field(default_factory=list,
                                        description="Why route was or was not authorized.")
    request_id: Optional[str] = None


def safe_default(request_text: str, reason: str, latency_ms: float = 0.0,
                 model_version: str = "fallback") -> Decision:
    """Conservative answer used whenever the model cannot be trusted or reached.

    It escalates rather than under-routes: a hard task handled by a strong agent costs tokens, the
    reverse costs correctness. Nothing here is auto-applied -- Codex keeps its own defaults.
    """
    return Decision(
        route="agent", agent="normal_code", reasoning="high", retrieval=True, clarify=False,
        confidence={k: 0.0 for k in ("route", "agent", "reasoning", "retrieval", "clarify")},
        latency_ms=round(latency_ms, 2), model_version=model_version, degraded=True, reason=reason,
        applied_policy={k: False for k in ("route", "agent", "reasoning", "retrieval", "clarify")},
        requires_human_approval=True,
        authorization_reasons=["router degraded; nothing may be applied automatically"],
    )
