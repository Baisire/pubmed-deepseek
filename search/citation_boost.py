"""引文增强与综合评分 - elink citedin + 多维评分融合。

职责：
- 通过 PubMed elink citedin 获取被引次数（缓存 7 天）
- 计算综合评分：语义相关性 + 时效 + 引文 + PubMed 排序
- 按综合评分排序文献列表

评分公式：
    score = 0.45 * rerank_score     # Cross-Encoder 语义相关性（主导）
          + 0.20 * recency_score    # 发表年份归一化
          + 0.20 * citation_score   # 被引次数归一化（log）
          + 0.15 * pubmed_score     # PubMed 原始排序归一化
"""

import json
import math
import os
import sqlite3
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Optional

import requests

from .models import Article

_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
_CACHE_DB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "pubmed_users.db",
)
_CITATION_CACHE_DAYS = 7

# 综合评分权重
W_RERANK = 0.45
W_RECENCY = 0.20
W_CITATION = 0.20
W_PUBMED = 0.15


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_CACHE_DB)
    conn.row_factory = sqlite3.Row
    return conn


def _init_citation_cache() -> None:
    with _get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS citation_cache (
                pmid TEXT PRIMARY KEY,
                citation_count INTEGER,
                source TEXT,
                cached_at TEXT DEFAULT (datetime('now', 'localtime'))
            )
            """
        )


def _get_cached_citation(pmid: str) -> Optional[int]:
    """获取缓存的被引次数（7 天内有效）。"""
    _init_citation_cache()
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT citation_count, cached_at FROM citation_cache WHERE pmid = ?",
            (pmid,),
        ).fetchone()
    if not row:
        return None
    # 检查是否过期
    try:
        cached_at = datetime.strptime(row["cached_at"], "%Y-%m-%d %H:%M:%S")
        age_days = (datetime.now() - cached_at).days
        if age_days > _CITATION_CACHE_DAYS:
            return None
    except ValueError:
        return None
    return row["citation_count"]


def _cache_citation(pmid: str, count: int, source: str = "elink") -> None:
    with _get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO citation_cache (pmid, citation_count, source) "
            "VALUES (?, ?, ?)",
            (pmid, count, source),
        )


def get_citation_counts(pmids: list[str], api_key: str = "") -> dict[str, int]:
    """通过 PubMed elink citedin 批量获取被引次数。

    Returns:
        {pmid: citation_count}
    """
    result: dict[str, int] = {}
    if not pmids:
        return result

    # 1. 查缓存
    uncached: list[str] = []
    for pmid in pmids:
        cached = _get_cached_citation(pmid)
        if cached is not None:
            result[pmid] = cached
        else:
            uncached.append(pmid)

    if not uncached:
        return result

    # 2. 调 elink citedin（一次最多 200 个 PMID）
    params = {
        "dbfrom": "pubmed",
        "db": "pubmed",
        "cmd": "neighbor",
        "linkname": "pubmed_pubmed_citedin",
    }
    for pmid in uncached:
        params.setdefault("id", []).append(pmid)
    if api_key:
        params["api_key"] = api_key

    try:
        resp = requests.get(f"{_EUTILS_BASE}elink.fcgi", params=params, timeout=30)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)

        # 解析：每个 LinkSet 对应一个输入 PMID
        for linkset in root.findall(".//LinkSet"):
            input_pmid = linkset.findtext("IdList/Id", "")
            if not input_pmid:
                continue

            count = 0
            for linksetdb in linkset.findall("LinkSetDb"):
                linkname = linksetdb.findtext("LinkName", "")
                if linkname == "pubmed_pubmed_citedin":
                    count = len(linksetdb.findall("Link"))
                    break

            result[input_pmid] = count
            _cache_citation(input_pmid, count, "elink")

    except (requests.RequestException, ET.ParseError):
        # elink 失败，未获取的 PMID 设为 0
        for pmid in uncached:
            if pmid not in result:
                result[pmid] = 0

    return result


def _recency_score(pub_date: str) -> float:
    """发表年份归一化分数（越新越高，0-1）。"""
    try:
        year = int(pub_date[:4])
        current_year = datetime.now().year
        # 线性归一化：5 年内 -> 0.8-1.0，10 年内 -> 0.5-0.8
        age = current_year - year
        if age <= 0:
            return 1.0
        elif age <= 5:
            return 1.0 - 0.04 * age
        elif age <= 10:
            return 0.8 - 0.06 * (age - 5)
        elif age <= 20:
            return 0.5 - 0.03 * (age - 10)
        else:
            return max(0.1, 0.2 - 0.01 * (age - 20))
    except (ValueError, IndexError):
        return 0.3  # 未知年份给中等偏低分


def _citation_score(count: int, max_count: int) -> float:
    """被引次数归一化（log 缩放，0-1）。"""
    if max_count <= 0:
        return 0.0
    return math.log(count + 1) / math.log(max_count + 1)


def _pubmed_score(rank: int, total: int) -> float:
    """PubMed 排序归一化（排越前越高，0-1）。"""
    if total <= 0:
        return 0.5
    return 1.0 - (rank / total)


def score_and_rank(articles: list[Article], api_key: str = "",
                   use_citations: bool = True) -> list[Article]:
    """计算综合评分并排序。

    Args:
        articles: 已经过语义精排的文献列表
        api_key: NCBI API Key
        use_citations: 是否获取引文数据（可关闭以降低延迟）

    Returns:
        按 final_score 降序排列的 Article 列表
    """
    if not articles:
        return []

    total = len(articles)

    # 1. 获取引文数据（可选）
    citation_counts: dict[str, int] = {}
    if use_citations:
        pmids = [a.pmid for a in articles]
        citation_counts = get_citation_counts(pmids, api_key=api_key)

    max_citations = max(citation_counts.values()) if citation_counts else 0

    # 2. 计算综合评分
    for a in articles:
        a.citation_count = citation_counts.get(a.pmid, 0)

        # 如果有 rerank_score 用 rerank，否则用 semantic_score
        relevance = a.rerank_score if a.rerank_score > 0 else a.semantic_score

        a.final_score = (
            W_RERANK * relevance
            + W_RECENCY * _recency_score(a.pub_date)
            + W_CITATION * _citation_score(a.citation_count, max_citations)
            + W_PUBMED * _pubmed_score(a.pubmed_rank, total)
        )

    # 3. 按综合评分排序
    articles.sort(key=lambda a: a.final_score, reverse=True)
    return articles
