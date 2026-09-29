"""
Internal HTTP API. The MCP server is a thin client of this, so the router can be tested,
benchmarked and used from anything that speaks HTTP, not only from Codex.

  POST /v1/route        the decision
  POST /v1/feedback     what actually happened afterwards
  GET  /health          process is alive (never fails while the process runs)
  GET  /ready           model is loaded and serving (for a load balancer / compose healthcheck)
  GET  /metrics         live counters, latency percentiles, VRAM
  GET  /v1/status       champion, challenger, canary, gates
  GET  /v1/policy       why each decision is or is not auto-appliable
  POST /v1/admin/reload re-read the registry after a promotion or rollback
  GET  /v1/review       what is waiting for human labelling
"""
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

from . import registry
from .schema import SCHEMA_VERSION, Decision, RouteRequest, SystemOneRequest
from .service import RouterService

logging.basicConfig(level=os.getenv("ROUTER_LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("router.api")
service = RouterService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        service.start()
    except Exception as e:                 # start anyway: /health and /ready must answer honestly
        logger.error("startup failed: %s", e)
        service.last_error = str(e)
    yield


app = FastAPI(title="Decizion", version=str(SCHEMA_VERSION), lifespan=lifespan)


class FeedbackRequest(BaseModel):
    request_id: str
    outcome: str        # ok | failed | escalated | wrong_agent | wrong_reasoning
    note: str = ""
    actual_agent: Optional[str] = None        # which agent really did the work
    actual_reasoning: Optional[str] = None    # which effort was really used
    needed_more_retrieval: Optional[bool] = None


@app.post("/v1/route", response_model=Decision)
async def route(req: RouteRequest) -> Decision:
    import asyncio
    return await asyncio.to_thread(service.route, req)


@app.post("/v1/systemone")
async def system_one(req: SystemOneRequest) -> Dict[str, Any]:
    """Evaluate dynamic typed questions with the selected local model."""
    import asyncio
    if not service.ready:
        raise HTTPException(503, "router not ready")
    try:
        return await asyncio.to_thread(service.system_one, req.state, req.questions, req.model)
    except Exception as e:
        logger.exception("typed decision inference failed")
        raise HTTPException(500, "typed decision inference failed") from e


@app.post("/v1/feedback")
async def feedback(req: FeedbackRequest) -> Dict[str, Any]:
    service.feedback(req.request_id, req.outcome, req.note, req.actual_agent,
                     req.actual_reasoning, req.needed_more_retrieval)
    return {"ok": True}


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok", "schema_version": SCHEMA_VERSION}


@app.get("/ready")
async def ready() -> JSONResponse:
    st = service.status()
    code = 200 if st["ready"] else 503
    return JSONResponse(status_code=code, content={"ready": st["ready"], "champion": st["champion"],
                                                   "last_error": st["last_error"]})


@app.get("/metrics")
async def metrics() -> Dict[str, Any]:
    st = service.status()
    m = service.metrics.snapshot()
    m.update({"vram_mb": st["vram_mb"], "champion": st["champion"], "challenger": st["challenger"],
              "canary_percent": st["canary_percent"], "max_len": st["max_len"],
              "review_queue": service.queue.counts()})
    return m


@app.get("/metrics.prom", response_class=PlainTextResponse)
async def metrics_prom() -> str:
    """Same numbers in Prometheus text format, for whatever scrapes it."""
    m = await metrics()
    out = []

    def emit(name: str, value: Any, labels: str = ""):
        if isinstance(value, (int, float)):
            out.append(f"router_{name}{labels} {value}")

    emit("requests_total", m["requests"])
    emit("degraded_total", m["degraded"])
    emit("degraded_rate", m["degraded_rate"])
    for p, v in (m["latency_ms"] or {}).items():
        emit("latency_ms", v, f'{{quantile="{p}"}}')
    for k, v in (m.get("route") or {}).items():
        emit("route_total", v, f'{{route="{k}"}}')
    for k, v in (m.get("agent") or {}).items():
        emit("agent_total", v, f'{{agent="{k}"}}')
    for k, v in (m.get("reasoning") or {}).items():
        emit("reasoning_total", v, f'{{reasoning="{k}"}}')
    emit("false_fast_path_rate", m["possible_false_fast_path_rate"])
    emit("agent_reasoning_disagreement_rate", m["agent_reasoning_disagreement_rate"])
    emit("vram_mb", m.get("vram_mb") or 0)
    return "\n".join(out) + "\n"


@app.get("/v1/status")
async def status() -> Dict[str, Any]:
    return service.status()


@app.get("/v1/policy")
async def policy() -> Dict[str, Any]:
    if service.policy is None:
        raise HTTPException(503, "not ready")
    return {"auto_apply": service.policy.applied, "detail": service.policy.explain(),
            "note": "A decision with auto_apply=false is advisory: Codex should keep its own default."}


@app.get("/v1/review")
async def review(limit: int = 50, priority: Optional[str] = None) -> Dict[str, Any]:
    """Cases awaiting a human label. Ask for `priority=P0` first: those are the expensive mistakes."""
    pend = service.queue.pending(priority)
    return {"pending": len(pend), "counts": service.queue.counts(),
            "priorities": {p: len(service.queue.pending(p)) for p in ("P0", "P1", "P2")},
            "items": pend[-limit:]}


@app.post("/v1/admin/reload")
async def reload_() -> Dict[str, Any]:
    return service.reload()


@app.post("/v1/admin/canary")
async def canary(percent: int) -> Dict[str, Any]:
    registry.set_canary(percent)
    return service.reload()


def main():
    import uvicorn
    uvicorn.run(app, host=os.getenv("ROUTER_HOST", "127.0.0.1"),
                port=int(os.getenv("ROUTER_PORT", "8099")), log_level="info")


if __name__ == "__main__":
    main()
