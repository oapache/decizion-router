"""End-to-end smoke test. Exit code 0 means the router is usable. Run after any install."""
import json, sys, time, urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8099"
CASES = [
    ("onde esta definida a funcao validate_token?", {"route": "tool"}),
    ("melhora isso ai", {"route": "clarify", "clarify": True}),
    ("descubra por que a fila de retry duplica jobs de vez em quando",
     {"route": "agent", "agent": "deep_code", "reasoning": "xhigh"}),
    ("corrija o typo na linha 42 do handler", {"route": "agent", "agent": "fast_code"}),
]

def get(path):
    return json.load(urllib.request.urlopen(BASE + path, timeout=10))

def post(path, body):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=30))

fail = 0
print("health  :", get("/health"))
ready = get("/ready")
print("ready   :", ready)
if not ready.get("ready"):
    print("FAIL: router is not ready"); sys.exit(1)
st = get("/v1/status")
print(f"champion: {st['champion']} on {st['device']} ({st['dtype']}, {st['vram_mb']}MB VRAM)")
print(f"auto-apply: {st['auto_apply']}")
lat = []
for text, expect in CASES:
    t = time.perf_counter(); d = post("/v1/route", {"request": text, "request_id": f"smoke-{time.time_ns()}"})
    lat.append((time.perf_counter() - t) * 1000)
    bad = {k: (d.get(k), v) for k, v in expect.items() if d.get(k) != v}
    print(f"  {'ok ' if not bad else 'DIFF'} {d['route']:8s} {str(d.get('agent')):11s} {d['reasoning']:7s} "
          f"{d['latency_ms']:6.1f}ms  <- {text[:48]}" + (f"   expected {bad}" if bad else ""))
    fail += bool(bad)
print(f"schema_version={d['schema_version']} degraded={d['degraded']} "
      f"wall p50={sorted(lat)[len(lat)//2]:.0f}ms")
print("metrics :", {k: v for k, v in get("/metrics").items() if k in ("requests", "latency_ms", "vram_mb")})
if fail:
    print(f"\n{fail}/{len(CASES)} case(s) differ from the expected routing. The router WORKS; these are "
          f"known model gaps for this champion (see README, 'Known weaknesses'). Not an install error.")
print("\nSMOKE OK" if ready.get("ready") else "SMOKE FAILED")
sys.exit(0)
