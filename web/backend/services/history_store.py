"""File-backed run history store for the web workbench."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Lock
from typing import Any

from analytics import NumpyEncoder


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class HistoryStore:
    """Persist run payloads and a compact index."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.runs_dir = self.root / "runs"
        self.index_path = self.root / "index.json"
        self._lock = Lock()
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        if not self.index_path.exists():
            self.index_path.write_text(
                json.dumps({"runs": []}, indent=2),
                encoding="utf-8",
            )

    def _load_index(self) -> dict[str, Any]:
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("runs", []), list):
                return data
            return {"runs": []}
        except Exception:
            return {"runs": []}

    def _save_index(self, index_data: dict[str, Any]) -> None:
        self.index_path.write_text(
            json.dumps(index_data, cls=NumpyEncoder, indent=2),
            encoding="utf-8",
        )

    def _run_path(self, run_id: str) -> Path:
        return self.runs_dir / f"{run_id}.json"

    def save_run(
        self,
        run_id: str,
        kind: str,
        status: str,
        payload: dict[str, Any],
        summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Write run payload and upsert index metadata."""
        run_path = self._run_path(run_id)
        now = _utc_now_iso()
        summary = summary or {}

        with self._lock:
            run_path.write_text(
                json.dumps(payload, cls=NumpyEncoder, indent=2),
                encoding="utf-8",
            )

            index_data = self._load_index()
            runs = index_data.setdefault("runs", [])
            existing = next((r for r in runs if r.get("run_id") == run_id), None)
            if existing is None:
                entry = {
                    "run_id": run_id,
                    "kind": kind,
                    "status": status,
                    "created_at": now,
                    "updated_at": now,
                    "summary": summary,
                    "path": str(run_path),
                }
                runs.append(entry)
            else:
                existing["status"] = status
                existing["updated_at"] = now
                existing["summary"] = summary
                existing["path"] = str(run_path)
                entry = existing

            # newest first
            runs.sort(key=lambda r: r.get("updated_at", ""), reverse=True)
            self._save_index(index_data)

        return entry

    def update_status(
        self,
        run_id: str,
        status: str,
        summary: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Update status/summary for an existing run."""
        with self._lock:
            index_data = self._load_index()
            runs = index_data.get("runs", [])
            existing = next((r for r in runs if r.get("run_id") == run_id), None)
            if existing is None:
                return None
            existing["status"] = status
            existing["updated_at"] = _utc_now_iso()
            if summary is not None:
                existing["summary"] = summary
            self._save_index(index_data)
            return existing

    def list_runs(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            index_data = self._load_index()
            runs = index_data.get("runs", [])
            return list(runs[:limit])

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        run_path = self._run_path(run_id)
        if not run_path.exists():
            return None
        return json.loads(run_path.read_text(encoding="utf-8"))
