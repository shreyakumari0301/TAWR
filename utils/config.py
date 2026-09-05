from pathlib import Path
from typing import Any

import yaml

from utils.paths import resolve_path


def load_config(path: str | Path) -> dict[str, Any]:
    with open(resolve_path(path), "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}
