"""
Make the router usable from Codex Desktop: check the pieces, then print what is left to do.

Run from the repo root:  python decizion-router/scripts/codex-setup.py
"""
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
OK, BAD, WARN = "  [ok]  ", "  [--]  ", "  [!!]  "
problems = []


def check(label, ok, detail="", fix=""):
    print((OK if ok else BAD) + label + (f"  {detail}" if detail else ""))
    if not ok:
        problems.append((label, fix))
    return ok


print("Decizion router -> Codex Desktop\n")

url = os.getenv("ROUTER_URL", "http://127.0.0.1:8099")
try:
    r = json.load(urllib.request.urlopen(url + "/v1/status", timeout=5))
    check("router responding", r["ready"], f"{r['champion']} on {r['device']} {r['dtype']}, "
                                           f"{r['vram_mb']}MB, cold start {r['load_seconds']}s")
    auto = [k for k, v in r["auto_apply"].items() if v]
    advisory = [k for k, v in r["auto_apply"].items() if not v]
    print(f"         auto-apply: {', '.join(auto) or 'none'}")
    print(f"         advisory  : {', '.join(advisory) or 'none'}  (Codex keeps its own default for these)")
except Exception as e:
    check("router responding", False, str(e)[:60],
          "start it:  cd decizion-router && docker compose up -d   (or ./run-router)")

check("mcp package importable", __import__("importlib").util.find_spec("mcp") is not None,
      sys.executable, "pip install mcp httpx")
global_codex_config = Path(os.getenv("CODEX_HOME", str(Path.home() / ".codex"))) / "config.toml"
config_text = global_codex_config.read_text(encoding="utf-8") if global_codex_config.exists() else ""
check("global Codex MCP registration", "[mcp_servers.decizion-router]" in config_text and str(REPO / "decizion-router" / "mcp_server.py").replace("\\", "/").lower() in config_text.lower(), str(global_codex_config), "register decizion-router in the global Codex config")
check("AGENTS.md", (REPO / "AGENTS.md").exists())

agents = sorted((REPO / ".codex/agents").glob("*.toml")) if (REPO / ".codex/agents").exists() else []
check("agent definitions", len(agents) == 3, f"{len(agents)} found")
missing_model = [p.name for p in agents if "# model =" in p.read_text(encoding="utf-8")]
if missing_model:
    print(WARN + f"model not set in: {', '.join(missing_model)}")
    problems.append(("agent model ids",
                     "open .codex/agents/*.toml and replace the commented `# model =` line with a "
                     "model your Codex install actually exposes"))

print("\nCodex will launch the MCP server as:")
print(f'  MCP server: decizion-router   script: {REPO / "decizion-router" / "mcp_server.py"}')
print("  tools: route_task, report_outcome, router_status")

if problems:
    print("\nLeft to do:")
    for label, fix in problems:
        print(f"  - {label}: {fix}")
    sys.exit(1)
print("\nReady. Restart Codex Desktop to load the renamed global MCP registration.")
