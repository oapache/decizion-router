"""
Risk and reversibility: what the router is allowed to authorize, as opposed to what it predicts.

Accuracy alone must never authorize an action. 86% route accuracy means roughly one request in
seven is routed wrong, which is fine for "should this be an agent or a tool call" and unacceptable
for "should this tool delete data". So authorization is a separate decision from prediction and it
combines four things:

    confidence  x  action class  x  risk signals in the request  x  reversibility

Read-only work can be authorized by a confident model. Anything that writes, deletes, migrates,
touches credentials, infrastructure or a database can never be authorized by model output alone,
however confident it is -- those keep Codex's own approval path, sandbox and permissions.

This layer only ever REMOVES authorization. It cannot grant more than the accuracy gates already
allow, and it never edits the prediction itself: `route` still says what the model thinks.
"""
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

# Action classes, from safest to most dangerous. A tool is classified by its id/description; when
# nothing matches we assume "unknown", which is treated as unsafe, not as safe.
READ_ONLY = "read_only"
WRITE = "write"
DESTRUCTIVE = "destructive"
UNKNOWN = "unknown"

TOOL_CLASS_PATTERNS = [
    (DESTRUCTIVE, r"\b(delete|remove|drop|purge|truncate|destroy|rm|revoke|rollback|reset|"
                  r"apagar?|remover?|excluir|deletar|force[_-]?push|migrate|migration)\b"),
    (WRITE, r"\b(write|create|update|edit|patch|commit|push|deploy|publish|send|post|insert|"
            r"upload|install|apply|merge|rename|move|set|configure|escrever|criar|alterar|"
            r"enviar|publicar|instalar)\b"),
    (READ_ONLY, r"\b(read|get|list|search|find|show|view|inspect|describe|query|grep|log|status|"
                r"diff|check|test|run[_-]?tests|analyz|ler|listar|buscar|procurar|mostrar|ver|"
                r"consultar|verificar)\b"),
]
_TOOL_CLASS = [(c, re.compile(p, re.I)) for c, p in TOOL_CLASS_PATTERNS]

# Risk signals in the request itself. These are independent of the tool: "apague os dados antigos"
# is dangerous even if no destructive tool was offered.
RISK_PATTERNS = {
    # \w* (zero or more), not \w+ (one or more): the bare word must match too. "duplica" alone
    # missed "concurrency" under \w+ because nothing followed it in "a fila duplica jobs" -- found
    # by checking a real router_status test against the live API instead of trusting the summary.
    "migration": r"\b(migra\w*|migrate|migration|backfill|schema|zero[- ]downtime|sem parar|"
                 r"uuid|alter table|drop column|reindex)\b",
    "data_loss": r"\b(perde\w*|perda|lost|losing|corromp\w*|corrupt\w*|apagar?|apague|delete[dr]?|"
                 r"drop|truncate|wipe|sumi\w*|vanish\w*|overwrite|sobrescrev\w*)\b",
    "concurrency": r"\b(race|corrida|concorr\w*|concurren\w*|deadlock|lock|duplica\w*|duplicat\w*|"
                   r"atomic|idempot\w*|thread[- ]?safe)\b",
    "security": r"\b(senha|password|credential|secret|token|api[_ -]?key|auth\w*|permiss\w*|"
                r"vulnerab\w*|inject\w*|cryptograph\w*|cifrad\w*|criptograf\w*|acesso root|sudo)\b",
    "infrastructure": r"\b(deploy|producao|production|kubernetes|k8s|terraform|infra\w*|cluster|"
                      r"dns|firewall|load balancer|escalar|scaling|restart the|reiniciar o servidor)\b",
    "database": r"\b(banco de dados|database|postgres|mysql|mongo|redis|sql|query|tabela|table|"
                r"indice|index|dump|restore)\b",
    "architecture": r"\b(arquitetura|architecture|refator\w+|refactor\w*|reestrutur\w+|"
                    r"restructur\w*|redesign|acoplament\w+|monolit\w+)\b",
}
_RISK = {k: re.compile(v, re.I) for k, v in RISK_PATTERNS.items()}

# Risks that can silently destroy or expose something. Never authorized by model confidence alone.
IRREVERSIBLE_RISKS = {"migration", "data_loss", "security", "infrastructure", "database"}


def risk_flags(text: str) -> List[str]:
    return [k for k, rx in _RISK.items() if rx.search(text or "")]


def classify_tool(tool_id: str, description: str = "") -> str:
    """Most dangerous class that matches wins: a tool called `sync_and_delete` is destructive."""
    blob = f"{tool_id} {description}"
    for cls, rx in _TOOL_CLASS:          # ordered destructive -> write -> read_only
        if rx.search(blob):
            return cls
    return UNKNOWN


def classify_tools(tools: Optional[Dict[str, str]]) -> Dict[str, str]:
    return {t: classify_tool(t, d) for t, d in (tools or {}).items()}


@dataclass
class Authorization:
    """What the caller may do automatically with this decision, and why."""
    route: bool = False
    reasons: List[str] = field(default_factory=list)
    risk_flags: List[str] = field(default_factory=list)
    irreversible: bool = False
    tool_class: str = UNKNOWN
    requires_human_approval: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {"risk_flags": self.risk_flags, "irreversible": self.irreversible,
                "tool_class": self.tool_class,
                "requires_human_approval": self.requires_human_approval,
                "reasons": self.reasons}


def authorize_route(decision_route: str, tool: Optional[str], tool_class: str,
                    route_confidence: float, flags: List[str], cfg: Dict[str, Any],
                    gate_passed: bool) -> Authorization:
    """Decide whether `route` may be applied without a human.

    Starts from the accuracy gate and takes authorization away; it never adds any.
    """
    a = Authorization(risk_flags=flags, tool_class=tool_class)
    min_conf_read = float(cfg.get("min_confidence_read_only", 0.90))
    a.irreversible = bool(set(flags) & IRREVERSIBLE_RISKS)

    if not gate_passed:
        a.reasons.append("route accuracy gate not met")
        return a
    if a.irreversible:
        # the expensive failure mode is silent and unrecoverable: a human decides, always
        a.reasons.append(f"irreversible risk in request ({', '.join(sorted(set(flags) & IRREVERSIBLE_RISKS))})")
        a.requires_human_approval = True
        return a
    if decision_route == "clarify":
        a.route = True            # asking the user is always safe
        a.reasons.append("clarify is a question, not an action")
        return a
    if decision_route == "agent":
        # choosing to involve an agent is not itself an action; what the agent then does stays
        # under Codex's own approval rules
        a.route = True
        a.reasons.append("delegating to an agent performs no action by itself")
        return a

    # route == tool: this one actually runs something
    if tool_class == READ_ONLY:
        if route_confidence >= min_conf_read:
            a.route = True
            a.reasons.append(f"read-only tool with confidence {route_confidence:.2f} >= {min_conf_read}")
        else:
            a.reasons.append(f"read-only tool but confidence {route_confidence:.2f} < {min_conf_read}")
        return a
    if tool_class in (WRITE, DESTRUCTIVE):
        a.reasons.append(f"{tool_class} tool is never authorized by model output alone")
        a.requires_human_approval = True
        return a
    a.reasons.append("tool class unknown; treated as unsafe")
    a.requires_human_approval = True
    return a
