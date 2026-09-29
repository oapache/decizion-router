"""
The fixed eval suite every candidate must face, unchanged, before it can be promoted.

Two sets, both versioned:
  evalset-vN   held-out phrasings of every archetype (never trained on)
  regression   real hard cases that once went wrong, in JSONL. A candidate that breaks one of these
               is rejected however good its averages look.

Average accuracy is not the target, so it is not the headline. What is reported per version:
  * accuracy per decision (route / agent / reasoning / retrieval / clarify)
  * for the two ordered decisions, errors UP and DOWN separately, plus the ordinal distance --
    xhigh -> high is one step, xhigh -> skip is three and is far worse
  * recall of the expensive classes: deep_code, xhigh, migration_risk
  * false_fast_path_rate: a risky request sent to tool or fast_code
  * agent_reasoning_disagreement_rate
  * ECE and Brier (is the confidence usable as a gate at all)
  * latency p50/p95/p99, VRAM, context length

Usage:
  python -m evals.suite --ckpt ../finetune/ckpt_ml_v7.pt --version router-v7
  python -m evals.suite --version router-v7 --from-registry
"""
import argparse
import json
import math
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO / "finetune"))

import torch  # noqa: E402

from router.decide import Policy, assemble, risk_flags, state_from  # noqa: E402
from router.model import RouterModel  # noqa: E402
from router.schema import AGENT_LEVELS, REASONING_LEVELS, RouteRequest  # noqa: E402

EVALSET_VERSION = "evalset-v1"
REGRESSION = ROOT / "evals" / "regression.jsonl"


# ---------------------------------------------------------------- the cases
def eval_cases(per_phrasing: int = 2, seed: int = 31337) -> List[Dict[str, Any]]:
    """Every held-out phrasing of every archetype, each sampled a few times with different noise."""
    import agent_routing as AR
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(next(
        Path.home().glob(".cache/huggingface/hub/models--convaiinnovations--laya/snapshots/*/multilingual/tokenizer"))))
    rng = random.Random(seed)
    out = []
    for name, (labels, templates) in AR.ARCHETYPES.items():
        evals = [t for t in templates if AR._in_split(t, "eval")] or templates
        for phrasing in evals:
            for _ in range(per_phrasing):
                ex = AR.make_example(rng, tok, split="eval", archetypes=[name], ask="route", balance=False)
                st = ex["state"]
                st["request"] = phrasing
                out.append({"source": "evalset", "archetype": name, "request": phrasing,
                            "project_context": st.get("project_context"),
                            "recent_turns": st.get("recent_turns"),
                            "labels": {"route": labels[0], "agent": labels[1], "reasoning": labels[2],
                                       "retrieval": labels[3] == "true", "clarify": labels[4] == "true"}})
    return out


def regression_cases() -> List[Dict[str, Any]]:
    """Real cases that were once wrong. Hand-labelled; a candidate must not break any of them."""
    if not REGRESSION.exists():
        return []
    out = []
    for line in REGRESSION.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        it = json.loads(line)
        it["source"] = "regression"
        it.setdefault("archetype", "regression")
        out.append(it)
    return out


# ---------------------------------------------------------------- metrics
def _ece(conf: List[float], correct: List[int], bins: int = 15) -> float:
    if not conf:
        return float("nan")
    n, e = len(conf), 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        sel = [j for j in range(n) if (conf[j] > lo or i == 0) and conf[j] <= hi]
        if sel:
            e += len(sel) / n * abs(statistics.mean(conf[j] for j in sel) - statistics.mean(correct[j] for j in sel))
    return round(e, 4)


def evaluate(model: RouterModel, policy: Policy, cases: List[Dict[str, Any]],
             warm: int = 5) -> Dict[str, Any]:
    for _ in range(warm):
        model.decide_raw({"request": "warmup"})

    rows, lat, ctx_len = [], [], []
    for c in cases:
        req = RouteRequest(request=c["request"], project_context=c.get("project_context"),
                           recent_turns=c.get("recent_turns"))
        t0 = time.perf_counter()
        parsed, ms = model.decide_raw(state_from(req))
        decision, audit = assemble(parsed, ms, model.version, policy, req.request)
        lat.append((time.perf_counter() - t0) * 1000)
        ctx_len.append(len(str(state_from(req))))
        labels = c["labels"]
        rows.append({"case": c, "archetype": c["archetype"], "source": c["source"],
                     "pred": {"route": decision.route, "agent": decision.agent,
                              "reasoning": decision.reasoning, "retrieval": decision.retrieval,
                              "clarify": decision.clarify},
                     "labels": labels, "conf": decision.confidence,
                     "raw": {k: parsed[k]["value"] for k in parsed},
                     "risk": audit.get("risk_flags", []),
                     "disagree": audit.get("agent_reasoning_disagreement", False)})

    m: Dict[str, Any] = {}
    for d in ("route", "agent", "reasoning", "retrieval", "clarify"):
        g = [r for r in rows if r["labels"].get(d) is not None and
             not (d == "agent" and r["labels"]["route"] != "agent")]
        if not g:
            continue
        ok = [int(r["pred"][d] == r["labels"][d]) for r in g]
        m[f"{d}_accuracy"] = round(statistics.mean(ok), 4)
        m[f"{d}_n"] = len(g)
        m[f"{d}_ece"] = _ece([r["conf"].get(d, 0.0) for r in g], ok)
        m[f"{d}_brier"] = round(statistics.mean(
            (1 - r["conf"].get(d, 0.0)) ** 2 if o else r["conf"].get(d, 0.0) ** 2
            for r, o in zip(g, ok)), 4)

    # ordered decisions: direction and distance of the error
    for d, levels in (("agent", AGENT_LEVELS), ("reasoning", REASONING_LEVELS)):
        g = [r for r in rows if r["labels"].get(d) in levels and r["pred"][d] in levels
             and not (d == "agent" and r["labels"]["route"] != "agent")]
        if not g:
            continue
        under = [r for r in g if levels.index(r["pred"][d]) < levels.index(r["labels"][d])]
        over = [r for r in g if levels.index(r["pred"][d]) > levels.index(r["labels"][d])]
        m[f"{d}_under_rate"] = round(len(under) / len(g), 4)
        m[f"{d}_over_rate"] = round(len(over) / len(g), 4)
        dist = [abs(levels.index(r["pred"][d]) - levels.index(r["labels"][d])) for r in g]
        m[f"{d}_mean_ordinal_distance"] = round(statistics.mean(dist), 4)
        m[f"{d}_severe_error_rate"] = round(sum(1 for x in dist if x >= 2) / len(g), 4)

    # recall of the expensive classes
    for name, d, value in (("deep_code_recall", "agent", "deep_code"),
                           ("xhigh_recall", "reasoning", "xhigh")):
        g = [r for r in rows if r["labels"].get(d) == value and
             not (d == "agent" and r["labels"]["route"] != "agent")]
        if g:
            m[name] = round(statistics.mean(int(r["pred"][d] == value) for r in g), 4)
            levels = AGENT_LEVELS if d == "agent" else REASONING_LEVELS
            miss = [levels.index(r["pred"][d]) for r in g if r["pred"][d] != value and r["pred"][d] in levels]
            m[name + "_miss_mean_level"] = round(statistics.mean(miss), 2) if miss else None
    g = [r for r in rows if r["archetype"] == "migration_risk"]
    if g:
        m["migration_risk_recall"] = round(statistics.mean(
            int(r["pred"].get("agent") == "deep_code" and r["pred"]["reasoning"] == "xhigh") for r in g), 4)

    # a risky request sent down the fast path is the expensive operational error -- but "risky" here
    # must come from the GOLDEN label, not from risk_flags(): risk.py pattern-matches the raw request
    # text and fires on topic words in read-only/informational requests too (e.g. "where do we define
    # the database schema for invoices?", "show me src/auth/middleware.ts", "what is the difference
    # between a mutex and a semaphore?" all trip migration/security/concurrency patterns despite being
    # correctly tool/fast_code). Using that as ground truth here punished the model for being CORRECT.
    # Nor is "reasoning in (high, xhigh)" enough on its own: fast_unfamiliar_api is genuinely
    # agent=fast_code with reasoning=high, so fast_code there is the right answer, not a fast-path
    # escape. The only thing that actually means "this needed more than the cheap path" is the golden
    # agent label itself calling for normal_code/deep_code.
    risky = [r for r in rows if r["labels"].get("route") == "agent" and r["labels"].get("agent") != "fast_code"]
    if risky:
        m["false_fast_path_rate"] = round(statistics.mean(
            int(r["pred"]["route"] == "tool" or r["pred"].get("agent") == "fast_code") for r in risky), 4)
    m["agent_reasoning_disagreement_rate"] = round(statistics.mean(int(r["disagree"]) for r in rows), 4)
    # and the opposite failure: everything shoved into deep/xhigh
    easy = [r for r in rows if r["labels"].get("agent") == "fast_code" or r["labels"].get("reasoning") in ("skip", "medium")]
    if easy:
        m["over_escalation_rate"] = round(statistics.mean(
            int(r["pred"].get("agent") == "deep_code" or r["pred"]["reasoning"] == "xhigh") for r in easy), 4)

    all_ok = [int(r["pred"][d] == r["labels"][d]) for r in rows
              for d in ("route", "agent", "reasoning", "retrieval", "clarify")
              if r["labels"].get(d) is not None and not (d == "agent" and r["labels"]["route"] != "agent")]
    m["overall_accuracy"] = round(statistics.mean(all_ok), 4)

    lat.sort()
    q = lambda p: round(lat[min(len(lat) - 1, int(len(lat) * p))], 2)
    m["latency_p50_ms"], m["latency_p95_ms"], m["latency_p99_ms"] = q(0.5), q(0.95), q(0.99)
    m["vram_mb"] = model.vram_mb()
    m["context_length_max"] = model.max_len
    m["context_chars_p95"] = int(sorted(ctx_len)[int(len(ctx_len) * 0.95)])
    m["n_cases"] = len(rows)

    # regression cases are pass/fail, never averaged away
    reg = [r for r in rows if r["source"] == "regression"]
    if reg:
        failed = [{"request": r["case"]["request"], "expected": r["labels"], "got": r["pred"]}
                  for r in reg if any(r["pred"].get(d) != r["labels"][d] for d in r["labels"]
                                      if d in ("route", "agent", "reasoning", "retrieval", "clarify")
                                      and not (d == "agent" and r["labels"]["route"] != "agent"))]
        m["regression_n"] = len(reg)
        m["regression_failed"] = len(failed)
        m["regression_pass"] = len(failed) == 0
        m["regression_failures"] = failed[:20]
    else:
        m["regression_n"] = 0
        m["regression_failed"] = 0
        m["regression_pass"] = True

    by_arch = defaultdict(list)
    for r in rows:
        by_arch[r["archetype"]].append(int(all(
            r["pred"][d] == r["labels"].get(d) for d in ("route", "reasoning") if d in r["labels"]
        )))
    m["weakest_archetypes"] = sorted(
        ({"archetype": a, "accuracy": round(statistics.mean(v), 3), "n": len(v)} for a, v in by_arch.items()),
        key=lambda x: x["accuracy"])[:6]
    return m, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--version", required=True)
    ap.add_argument("--from-registry", action="store_true")
    ap.add_argument("--base", default="multilingual")
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--per_phrasing", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--register", action="store_true",
                    help="write these metrics into the registry entry so the policy gates can read them")
    args = ap.parse_args()

    from router import registry
    if args.from_registry:
        mv = registry.get(args.version)
        assert mv, f"unknown version {args.version}"
        from router.model import build
        model = build(mv)
    else:
        assert args.ckpt, "--ckpt or --from-registry"
        model = RouterModel(checkpoint=args.ckpt, base_model=args.base, dtype=args.dtype,
                            max_len=args.max_len, version=args.version)
    model.load()
    from router.service import load_config
    policy = Policy(load_config(), {})       # gates off during measurement, so nothing is masked
    cases = eval_cases(args.per_phrasing) + regression_cases()
    print(f"{len(cases)} cases ({sum(1 for c in cases if c['source']=='regression')} regression)", flush=True)
    m, rows = evaluate(model, policy, cases)
    m["evalset_version"] = EVALSET_VERSION
    m["version"] = args.version
    out = Path(args.out or (ROOT / "evals" / "results" / f"{args.version}.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"metrics": m, "rows": [
        {k: r[k] for k in ("archetype", "source", "pred", "labels", "conf")} for r in rows]},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in m.items() if k != "regression_failures"}, indent=2, ensure_ascii=False))
    if args.register:
        mv = registry.get(args.version)
        assert mv, f"unknown version {args.version}; register it first"
        mv.metrics = {k: v for k, v in m.items() if not isinstance(v, (list, dict))}
        mv.evalset_version = EVALSET_VERSION
        registry.register(mv)
        print("metrics written into registry entry", args.version)
    print("saved", out)


if __name__ == "__main__":
    main()
