"""
Model registry: which checkpoint is champion, which is challenger, and how one replaces the other.

Everything a version needs to be reproduced is recorded together -- checkpoint, the dataset it was
trained on, the eval set it was measured against, its config and its metrics -- so a `.pt` file is
never an orphan. Nothing is promoted by finishing a training run: promotion is an explicit gated
decision, recorded with the numbers that justified it, and always reversible.
"""
import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "registry"
MODELS = REGISTRY / "models"
STATE = REGISTRY / "state.json"


@dataclass
class ModelVersion:
    version: str                      # router-v7
    checkpoint: str                   # path relative to the repo root
    base_model: str = "multilingual"  # Laya checkpoint it was fine-tuned from
    dataset_version: str = ""         # dataset-v7
    evalset_version: str = ""         # evalset-v3
    questions_version: int = 1        # config/questions-vN.json
    config: str = ""                  # config-v7.yaml
    dtype: str = "fp16"
    max_len: int = 1024
    created_at: float = field(default_factory=time.time)
    train_log: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""


def _load() -> Dict[str, Any]:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"champion": None, "challenger": None, "canary_percent": 0, "history": []}


def _save(state: Dict[str, Any]) -> None:
    REGISTRY.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def register(mv: ModelVersion) -> Path:
    """Record a version. Does not make it live."""
    MODELS.mkdir(parents=True, exist_ok=True)
    path = MODELS / f"{mv.version}.json"
    path.write_text(json.dumps(asdict(mv), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def get(version: str) -> Optional[ModelVersion]:
    path = MODELS / f"{version}.json"
    if not path.exists():
        return None
    return ModelVersion(**json.loads(path.read_text(encoding="utf-8")))


def list_versions() -> List[str]:
    return sorted(p.stem for p in MODELS.glob("*.json"))


def champion() -> Optional[ModelVersion]:
    v = _load().get("champion")
    return get(v) if v else None


def challenger() -> Optional[ModelVersion]:
    v = _load().get("challenger")
    return get(v) if v else None


def canary_percent() -> int:
    return int(_load().get("canary_percent", 0))


def set_challenger(version: str, canary_percent: int = 0) -> None:
    assert get(version), f"unknown version {version}"
    st = _load()
    st["challenger"] = version
    st["canary_percent"] = max(0, min(100, canary_percent))
    st["history"].append({"at": time.time(), "action": "set_challenger",
                          "version": version, "canary_percent": st["canary_percent"]})
    _save(st)


def set_canary(percent: int) -> None:
    st = _load()
    assert st.get("challenger"), "no challenger to send traffic to"
    st["canary_percent"] = max(0, min(100, percent))
    st["history"].append({"at": time.time(), "action": "set_canary", "canary_percent": st["canary_percent"]})
    _save(st)


def promote(version: str, report: Optional[Dict[str, Any]] = None) -> None:
    """Make `version` the champion. The previous champion stays registered for rollback."""
    assert get(version), f"unknown version {version}"
    st = _load()
    previous = st.get("champion")
    st["champion"] = version
    if st.get("challenger") == version:
        st["challenger"] = None
        st["canary_percent"] = 0
    st["history"].append({"at": time.time(), "action": "promote", "version": version,
                          "previous": previous, "report": report or {}})
    _save(st)


def reject(version: str, report: Optional[Dict[str, Any]] = None) -> None:
    st = _load()
    if st.get("challenger") == version:
        st["challenger"] = None
        st["canary_percent"] = 0
    st["history"].append({"at": time.time(), "action": "reject", "version": version, "report": report or {}})
    _save(st)


def rollback(to_version: Optional[str] = None) -> str:
    """Return to the previous approved champion (or an explicit one). Always available."""
    st = _load()
    if to_version is None:
        promotions = [h for h in st["history"] if h["action"] == "promote"]
        assert len(promotions) >= 2, "no earlier champion to roll back to"
        to_version = promotions[-1]["previous"] or promotions[-2]["version"]
    assert get(to_version), f"unknown version {to_version}"
    st["champion"] = to_version
    st["challenger"] = None
    st["canary_percent"] = 0
    st["history"].append({"at": time.time(), "action": "rollback", "version": to_version})
    _save(st)
    return to_version


def history(limit: int = 20) -> List[Dict[str, Any]]:
    return _load()["history"][-limit:]


def status() -> Dict[str, Any]:
    st = _load()
    return {"champion": st.get("champion"), "challenger": st.get("challenger"),
            "canary_percent": st.get("canary_percent", 0), "registered": list_versions()}
