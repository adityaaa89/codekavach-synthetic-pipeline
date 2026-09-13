from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
REFERENCE_DIR = ROOT / "data" / "reference"
GENERATED_DIR = ROOT / "data" / "generated"
ML_READY_DIR = ROOT / "data" / "ml_ready"
REPORTS_DIR = ROOT / "reports"


def ensure_dirs() -> None:
    for path in [CONFIG_DIR, REFERENCE_DIR, GENERATED_DIR, ML_READY_DIR, REPORTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
