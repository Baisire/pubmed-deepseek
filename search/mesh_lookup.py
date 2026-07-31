"""MeSH 术语查找与验证 - NLM MeSH REST API + SQLite 缓存。

职责：
- 验证候选词是否为有效 MeSH 术语
- 获取 MeSH 术语的 D 编号、树号、入口词、子树限定符
- 结果持久化到 SQLite（MeSH 年度更新，缓存可长期有效）
- 并行验证多个候选词

API:
    GET https://id.nlm.nih.gov/mesh/lookup/descriptor?label={term}&max=1
    -> [{"resource": "http://id.nlm.nih.gov/mesh/D009504", "label": "Neutropenia"}]

    GET https://id.nlm.nih.gov/mesh/{D编号}.json
    -> {identifier, label, treeNumber, allowableQualifier, concept, ...}
"""

import json
import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from urllib.parse import quote

import requests

from .models import MeshTerm

_MESH_API_BASE = "https://id.nlm.nih.gov/mesh"
_CACHE_DB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "pubmed_users.db",
)
_write_lock = threading.Lock()
_wal_initialized = False


def _get_conn() -> sqlite3.Connection:
    global _wal_initialized
    conn = sqlite3.connect(_CACHE_DB)
    conn.row_factory = sqlite3.Row
    if not _wal_initialized:
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            _wal_initialized = True
        except Exception:
            pass
    return conn


def _init_cache() -> None:
    """创建 mesh_cache 表（如不存在）。"""
    with _get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mesh_cache (
                term TEXT PRIMARY KEY,
                ui TEXT,
                tree_numbers TEXT,
                entry_terms TEXT,
                subheadings TEXT,
                cached_at TEXT DEFAULT (datetime('now', 'localtime'))
            )
            """
        )


def _cache_mesh(term: str, ui: str, tree_numbers: list[str],
                entry_terms: list[str], subheadings: list[str]) -> None:
    with _write_lock:
        with _get_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO mesh_cache (term, ui, tree_numbers, entry_terms, subheadings) "
                "VALUES (?, ?, ?, ?, ?)",
                (term, ui, json.dumps(tree_numbers), json.dumps(entry_terms),
                 json.dumps(subheadings)),
            )


def _get_cached_mesh(term: str) -> Optional[MeshTerm]:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM mesh_cache WHERE term = ?", (term,)
        ).fetchone()
    if not row:
        return None
    return MeshTerm(
        term=term,
        ui=row["ui"],
        tree_numbers=json.loads(row["tree_numbers"]),
        entry_terms=json.loads(row["entry_terms"]),
        subheadings=json.loads(row["subheadings"]),
    )


def lookup_mesh_term(term: str, timeout: int = 10) -> Optional[MeshTerm]:
    """验证一个 MeSH 候选词，返回精确信息或 None。

    优先查 SQLite 缓存，未命中则调 MeSH REST API。
    """
    _init_cache()

    # 1. 查缓存
    cached = _get_cached_mesh(term)
    if cached is not None:
        return cached

    # 2. 调 API: 查 descriptor
    try:
        resp = requests.get(
            f"{_MESH_API_BASE}/lookup/descriptor",
            params={"label": term, "max": 1},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return None

        resource_uri = data[0].get("resource", "")
        # 从 resource URI 提取 UI（如 http://id.nlm.nih.gov/mesh/D009504 -> D009504）
        ui = resource_uri.rstrip("/").split("/")[-1]
        if not ui.startswith("D"):
            return None

    except (requests.RequestException, IndexError, KeyError):
        return None

    # 3. 调 API: 取详情
    try:
        detail_resp = requests.get(
            f"{_MESH_API_BASE}/{ui}.json",
            timeout=timeout,
        )
        detail_resp.raise_for_status()
        detail = detail_resp.json()

        tree_numbers = [t.get("@value", "") for t in detail.get("treeNumber", [])
                        if isinstance(t, dict)]
        if not tree_numbers and isinstance(detail.get("treeNumber"), str):
            tree_numbers = [detail["treeNumber"]]

        # 入口词从 concept 中提取
        entry_terms = []
        for concept in detail.get("concept", []):
            if isinstance(concept, dict):
                for term_entry in concept.get("term", []):
                    if isinstance(term_entry, dict):
                        entry_terms.append(term_entry.get("@value", ""))

        subheadings = []
        for q in detail.get("allowableQualifier", []):
            if isinstance(q, dict):
                subheadings.append(q.get("@value", ""))

        mesh = MeshTerm(
            term=detail.get("label", {}).get("@value", term),
            ui=ui,
            tree_numbers=tree_numbers,
            entry_terms=[e for e in entry_terms if e],
            subheadings=[s for s in subheadings if s],
        )

        _cache_mesh(mesh.term, mesh.ui, mesh.tree_numbers,
                    mesh.entry_terms, mesh.subheadings)
        return mesh

    except (requests.RequestException, KeyError, TypeError):
        return None


def verify_mesh_batch(candidates: list[str], max_workers: int = 3) -> dict[str, Optional[MeshTerm]]:
    """并行验证多个 MeSH 候选词。

    Returns:
        {candidate_term: MeshTerm or None}
    """
    results: dict[str, Optional[MeshTerm]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(lookup_mesh_term, c): c for c in candidates}
        for future in as_completed(futures):
            candidate = futures[future]
            try:
                results[candidate] = future.result()
            except Exception:
                results[candidate] = None
    return results
