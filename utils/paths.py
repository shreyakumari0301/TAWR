from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
