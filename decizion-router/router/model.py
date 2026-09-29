"""
The router model: one Laya checkpoint, loaded once, resident on the GPU in fp16.

Loading is the expensive part (tens of seconds), inference is ~50ms, so the process holds the model
for its whole life and never reloads per request. Two models can be resident at once (champion and
challenger) so a canary costs latency, not a reload.
"""
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from .schema import AGENT_LEVELS, REASONING_LEVELS

logger = logging.getLogger("router.model")
ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parent


def load_questions(version: int = 1) -> Dict[str, Any]:
    """The exact question wording the checkpoint was trained with -- a versioned artifact.

    Laya reads the option descriptions as part of its input, so changing a single word here changes
    the model's behaviour. It is frozen per version rather than imported from the training code.
    """
    path = ROOT / "config" / f"questions-v{version}.json"
    return json.loads(path.read_text(encoding="utf-8"))["decisions"]


class RouterModel:
    def __init__(self, checkpoint: str, base_model: str = "multilingual", dtype: str = "fp16",
                 max_len: int = 1024, questions_version: int = 1, version: str = "unknown",
                 device: str = "cuda"):
        self.version = version
        self.checkpoint = checkpoint
        self.max_len = max_len
        self.questions = load_questions(questions_version)
        self._lock = threading.Lock()   # Laya's forward is not re-entrant on one agent instance
        self.loaded_at: Optional[float] = None
        self.load_seconds: Optional[float] = None
        self.device = device
        self._agent = None
        self._base_model = base_model
        self._dtype = dtype

    # -- lifecycle ----------------------------------------------------------
    def load(self) -> None:
        import laya
        t0 = time.perf_counter()
        want_cuda = self.device == "cuda" and torch.cuda.is_available()
        if self.device == "cuda" and not want_cuda:
            logger.warning("CUDA requested but unavailable; the router will run on CPU (slower).")
        dev = "cuda" if want_cuda else "cpu"
        agent = laya.Router(device=dev, max_loaded=1).load(self._base_model)
        ckpt = Path(self.checkpoint)
        if not ckpt.is_absolute():
            ckpt = REPO / ckpt
        if not ckpt.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt}")
        agent.model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
        agent.model.to(agent.device).eval()
        if want_cuda and self._dtype in ("fp16", "bf16"):
            dt = torch.float16 if self._dtype == "fp16" else torch.bfloat16
            agent.model.encoder.to(dt)
            agent.dtype = dt  # autocast dtype must match the stored weights, or they are re-cast per call
        agent.cfg["max_len"] = self.max_len
        self._agent = agent
        self.device = dev
        # warm up: the first forward pays cuDNN autotuning, later calls must not
        for _ in range(3):
            self._raw({"request": "warmup"}, list(self.questions))
        self.load_seconds = time.perf_counter() - t0
        self.loaded_at = time.time()
        logger.info("%s loaded on %s in %.1fs (dtype=%s, max_len=%d)",
                    self.version, dev, self.load_seconds, self._dtype, self.max_len)

    @property
    def ready(self) -> bool:
        return self._agent is not None

    def vram_mb(self) -> Optional[float]:
        if self.device != "cuda" or not torch.cuda.is_available():
            return None
        return round(torch.cuda.memory_allocated() / 2 ** 20, 1)

    def count_tokens(self, state: Dict[str, Any]) -> Optional[int]:
        """Tokens the model actually sees for this state, before its own max_len truncation."""
        if self._agent is None:
            return None
        try:
            from laya.common import serialize_state
            return len(self._agent.tok(serialize_state(state), add_special_tokens=False)["input_ids"])
        except Exception:
            return None

    # -- inference ----------------------------------------------------------
    def _raw(self, state: Dict[str, Any], asks: List[str],
             candidate_tools: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        qs = {}
        for a in asks:
            q = dict(self.questions[a])
            q.pop("levels", None)
            qs[a] = q
        if candidate_tools:
            qs["tool"] = {"type": "choice",
                          "instructions": "Qual ferramenta atende o pedido mais recente do usuario?",
                          "criteria": dict(candidate_tools)}
        with self._lock:
            return self._agent.system_one(state, qs)["answers"]

    def decide_raw(self, state: Dict[str, Any],
                   candidate_tools: Optional[Dict[str, str]] = None) -> Tuple[Dict[str, Any], float]:
        """All five decisions in one forward pass. Returns (parsed answers, latency in ms)."""
        if not self.ready:
            raise RuntimeError("model not loaded")
        t0 = time.perf_counter()
        raw = self._raw(state, list(self.questions), candidate_tools)
        ms = (time.perf_counter() - t0) * 1000
        return self._parse(raw), ms

    def system_one(self, state: Any, questions: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Any], float]:
        """Run caller-supplied typed questions in one pass on this resident model."""
        if not self.ready:
            raise RuntimeError("model not loaded")
        t0 = time.perf_counter()
        with self._lock:
            result = self._agent.system_one(state, questions)
        return result, (time.perf_counter() - t0) * 1000

    @staticmethod
    def _parse(raw: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, a in raw.items():
            if a["type"] == "choice":
                out[key] = {"value": a["choice"], "confidence": a["confidence"],
                            "probabilities": a["probabilities"]}
            elif a["type"] == "noul":
                out[key] = {"value": a["noul"] > 0.5, "confidence": a["confidence"], "p_true": a["noul"]}
            else:  # score
                idx = min(len(REASONING_LEVELS) - 1, max(0, int(round(a["score"]))))
                out[key] = {"value": REASONING_LEVELS[idx], "confidence": a["confidence"],
                            "score": a["score"], "probabilities": a.get("probabilities", {})}
        return out


def build(mv, device: str = "cuda") -> RouterModel:
    """RouterModel from a registry entry."""
    return RouterModel(checkpoint=mv.checkpoint, base_model=mv.base_model, dtype=mv.dtype,
                       max_len=mv.max_len, questions_version=mv.questions_version,
                       version=mv.version, device=device)
