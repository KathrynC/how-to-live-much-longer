"""Index and search local Zotero PDF storage."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def _hash_file(path: Path, chunk: int = 1 << 20) -> str:
    """Return a short sha1 for a file."""
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def _title_from_name(name: str) -> str:
    """Heuristic: strip extension and normalize spaces."""
    stem = Path(name).stem
    return " ".join(stem.replace("_", " ").split())


def build_index(storage: Path) -> list[dict]:
    """Build index entries for PDFs under Zotero storage."""
    entries = []
    for path in storage.rglob("*.pdf"):
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        entries.append(
            {
                "path": str(path),
                "title_guess": _title_from_name(path.name),
                "size_bytes": stat.st_size,
                "mtime": stat.st_mtime,
                "sha1": _hash_file(path),
            }
        )
    return entries


def search_index(entries: list[dict], query: str) -> list[dict]:
    """Return entries where query matches path or title."""
    q = query.lower()
    out = []
    for entry in entries:
        if q in entry.get("title_guess", "").lower() or q in entry.get("path", "").lower():
            out.append(entry)
    return out


def main() -> None:
    """CLI for indexing and searching Zotero storage."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--storage", default=os.environ.get("ZOTERO_STORAGE", str(Path("~/Zotero/storage").expanduser())))
    ap.add_argument("--out", default="artifacts/zotero_index.json")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--query", default=None)
    args = ap.parse_args()

    storage = Path(args.storage).expanduser()
    if not storage.exists():
        raise SystemExit(f"Zotero storage not found: {storage}")

    out_path = Path(args.out)
    entries = []
    if out_path.exists() and not args.refresh:
        entries = json.loads(out_path.read_text())
    else:
        entries = build_index(storage)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(entries, indent=2))
        print(f"Wrote index: {out_path} ({len(entries)} PDFs)")

    if args.query:
        matches = search_index(entries, args.query)
        for entry in matches:
            print(
                f"{entry['title_guess']}\n"
                f"  {entry['path']}\n"
                f"  {entry['size_bytes'] / 1024 / 1024:.2f} MB\n"
                f"  sha1={entry['sha1'][:10]}\n"
            )
        print(f"Matches: {len(matches)}")


if __name__ == "__main__":
    main()
