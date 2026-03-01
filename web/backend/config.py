"""Backend configuration for the web workbench."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from constants import OLLAMA_URL


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    """Runtime settings."""
    repo_root: Path = REPO_ROOT
    web_runs_root: Path = REPO_ROOT / "output" / "web_runs"
    ollama_url: str = OLLAMA_URL
    default_allow_origins: tuple[str, ...] = (
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    )


def get_settings() -> Settings:
    root_override = os.getenv("WEB_RUNS_ROOT")
    ollama_url = os.getenv("OLLAMA_URL", OLLAMA_URL)
    settings = Settings(ollama_url=ollama_url)
    if root_override:
        return Settings(
            repo_root=settings.repo_root,
            web_runs_root=Path(root_override).expanduser().resolve(),
            ollama_url=ollama_url,
            default_allow_origins=settings.default_allow_origins,
        )
    return settings
