from pathlib import Path


def repository_root() -> Path:
    """Return the repository root from the installed src-layout package."""
    return Path(__file__).resolve().parents[2]


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repository_root() / candidate
