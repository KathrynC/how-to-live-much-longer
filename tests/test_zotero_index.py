"""Tests for Zotero local index tooling."""
from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "zotero_index.py"
    spec = importlib.util.spec_from_file_location("zotero_index", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_index_and_search(tmp_path):
    module = _load_module()
    storage = tmp_path / "storage" / "ABCD1234"
    storage.mkdir(parents=True)
    pdf = storage / "Mito_Dynamics_Review.pdf"
    pdf.write_bytes(b"%PDF-1.4\nfake-pdf-content\n")

    entries = module.build_index(tmp_path / "storage")
    assert len(entries) == 1
    entry = entries[0]

    assert entry["title_guess"] == "Mito Dynamics Review"
    assert entry["path"].endswith("Mito_Dynamics_Review.pdf")
    assert entry["size_bytes"] == len(b"%PDF-1.4\nfake-pdf-content\n")
    assert len(entry["sha1"]) == 40

    matches = module.search_index(entries, "dynamics")
    assert len(matches) == 1
    assert matches[0]["path"] == entry["path"]


def test_title_normalization():
    module = _load_module()
    assert module._title_from_name("A__B   C.pdf") == "A B C"
