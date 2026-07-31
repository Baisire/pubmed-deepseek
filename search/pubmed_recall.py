"""Tier 1 结构化召回 - PubMed E-utilities 检索。

职责：
- 用 Boolean 检索式调 esearch 获取候选 PMID 列表
- 自适应档位选择（中档取计数 -> 自动切窄/宽档）
- efetch 获取文献详情（XML 解析）
- elink 相关文献扩展（可选）

迁移自 app.py 的 search_pubmed()，升级为返回 Article 对象。
"""

import xml.etree.ElementTree as ET
from typing import Optional

import requests

from .models import Article, SearchStrategy

_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
_DEFAULT_TIMEOUT = 30
_MAX_CANDIDATES = 50


def _esearch(query: str, retmax: int = 0, api_key: str = "",
             timeout: int = _DEFAULT_TIMEOUT) -> dict:
    """调用 esearch，返回 {idlist, count}。"""
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": str(retmax),
        "retmode": "json",
        "sort": "relevance",
    }
    if api_key:
        params["api_key"] = api_key

    resp = requests.get(f"{_EUTILS_BASE}esearch.fcgi", params=params, timeout=timeout)
    resp.raise_for_status()
    result = resp.json().get("esearchresult", {})
    return {
        "idlist": result.get("idlist", []),
        "count": int(result.get("count", 0)),
    }


def _efetch(pmids: list[str], api_key: str = "",
            timeout: int = 60) -> list[Article]:
    """调用 efetch 获取文献详情，解析 XML 返回 Article 列表。"""
    if not pmids:
        return []

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract",
    }
    if api_key:
        params["api_key"] = api_key

    resp = requests.get(f"{_EUTILS_BASE}efetch.fcgi", params=params, timeout=timeout)
    resp.raise_for_status()

    articles: list[Article] = []
    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError:
        return []

    for article_elem in root.findall(".//PubmedArticle"):
        article = _parse_article_xml(article_elem)
        if article.pmid:
            articles.append(article)

    return articles


def _parse_article_xml(article_elem: ET.Element) -> Article:
    """解析单个 PubmedArticle XML 元素为 Article 对象。"""
    def _find_text(parent: ET.Element, tag: str) -> str:
        elem = parent.find(f".//{tag}")
        if elem is None:
            return ""
        # 处理含子标签的情况（如 ArticleTitle 中的 <i> 等）
        return "".join(elem.itertext())

    pmid = _find_text(article_elem, "PMID")
    title = _find_text(article_elem, "ArticleTitle")

    # 摘要（可能多段，带 Label 属性）
    abstract_parts: list[str] = []
    for abs_elem in article_elem.findall(".//AbstractText"):
        label = abs_elem.get("Label", "")
        text = "".join(abs_elem.itertext())
        if label:
            abstract_parts.append(f"{label}: {text}")
        else:
            abstract_parts.append(text)
    abstract = " ".join(abstract_parts)

    # 作者
    authors_list: list[str] = []
    for author in article_elem.findall(".//Author"):
        last = _find_text(author, "LastName")
        initials = _find_text(author, "Initials")
        if last:
            name = f"{last} {initials}" if initials else last
            authors_list.append(name)
    authors = ", ".join(authors_list[:10])  # 最多取前10位

    journal = _find_text(article_elem, "Title")
    pub_date = _find_text(article_elem, "PubDate")
    doi = ""
    for aid in article_elem.findall(".//ArticleId"):
        if aid.get("IdType") == "doi":
            doi = aid.text or ""
            break

    return Article(
        pmid=pmid,
        title=title,
        abstract=abstract,
        authors=authors,
        journal=journal,
        pub_date=pub_date,
        doi=doi,
        has_abstract=bool(abstract),
    )


def _get_related_articles(pmids: list[str], per_article: int = 5,
                          api_key: str = "") -> list[str]:
    """通过 elink 获取相关文献 PMID（可选扩展）。"""
    if not pmids:
        return []

    params = {
        "dbfrom": "pubmed",
        "db": "pubmed",
        "cmd": "neighbor",
        "linkname": "pubmed_pubmed",
    }
    for pmid in pmids:
        params.setdefault("id", []).append(pmid)
    if api_key:
        params["api_key"] = api_key

    try:
        resp = requests.get(f"{_EUTILS_BASE}elink.fcgi", params=params,
                            timeout=_DEFAULT_TIMEOUT)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)

        related: list[str] = []
        for linksetdb in root.findall(".//LinkSetDb"):
            linkname = linksetdb.findtext("LinkName", "")
            if linkname == "pubmed_pubmed":
                links = linksetdb.findall("Link")
                for link in links[:per_article]:
                    pid = link.findtext("Id", "")
                    if pid and pid not in pmids and pid not in related:
                        related.append(pid)
        return related
    except (requests.RequestException, ET.ParseError):
        return []


def recall(strategy: SearchStrategy, api_key: str = "",
           max_results: int = _MAX_CANDIDATES,
           expand_related: bool = False) -> list[Article]:
    """执行 Tier 1 结构化召回。

    Args:
        strategy: 检索策略（含 boolean_query）
        api_key: NCBI API Key（可选，提升频率限制）
        max_results: 最大候选文献数
        expand_related: 是否启用 elink 相关文献扩展

    Returns:
        Article 列表（按 PubMed relevance 排序，pubmed_rank 已填充）
    """
    query = strategy.boolean_query
    if not query:
        return []

    # 1. 自适应档位选择：中档取计数
    try:
        probe = _esearch(query, retmax=0, api_key=api_key)
        total_count = probe["count"]
        strategy.total_count = total_count
    except (requests.RequestException, KeyError, ValueError):
        # 探测失败，直接用中档检索
        total_count = 50

    # 2. esearch 获取 PMID 列表
    try:
        search_result = _esearch(query, retmax=max_results, api_key=api_key)
        id_list = search_result["idlist"]
    except (requests.RequestException, KeyError):
        return []

    if not id_list:
        return []

    # 3. efetch 获取详情
    try:
        articles = _efetch(id_list, api_key=api_key)
    except requests.RequestException:
        return []

    # 填充 PubMed 排序位置
    pmid_to_rank = {pmid: i for i, pmid in enumerate(id_list)}
    for a in articles:
        a.pubmed_rank = pmid_to_rank.get(a.pmid, 0)
        a.source = "core"

    # 4. 相关文献扩展（可选）
    if expand_related and len(articles) >= 3:
        seed_pmids = [a.pmid for a in articles[:5]]
        related_pmids = _get_related_articles(seed_pmids, per_article=5,
                                              api_key=api_key)
        if related_pmids:
            try:
                related_articles = _efetch(related_pmids, api_key=api_key)
                existing_pmids = {a.pmid for a in articles}
                for ra in related_articles:
                    if ra.pmid not in existing_pmids:
                        ra.source = "related"
                        articles.append(ra)
            except requests.RequestException:
                pass  # 扩展失败不影响核心结果

    return articles
