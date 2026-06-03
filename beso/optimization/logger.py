"""Structured JSONL tracing for the BESO optimization loop (TICKET-001).

Terminal logs are insufficient for offline diagnostic review of the decision
matrix. ``JSONLLogger`` appends one structured JSON record per optimizer
iteration so the full optimization curve (pool edits, surrogate predictions,
acquisition scores, gate decisions, archive tiers) can be analyzed after a run.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class JSONLLogger:
    """Append-only JSON-lines writer for per-iteration optimizer traces."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.parent and not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, payload: dict[str, Any]) -> None:
        """Serialize and append a single nested payload as one JSON line."""

        line = json.dumps(payload, default=_json_default, ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def _json_default(obj: Any) -> Any:
    """Fallback serializer for dataclasses, enums, numpy types, and sets."""

    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=str)
    return str(obj)


__all__ = ["JSONLLogger"]
