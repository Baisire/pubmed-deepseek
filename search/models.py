"""数据模型 - v3.0 混合语义检索。

定义检索流程中各阶段传递的数据结构。
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MeshTerm:
    """MeSH 术语信息（经 REST API 验证）。"""

    term: str                    # 精确 MeSH 名称
    ui: str                      # D 编号，如 D009504
    tree_numbers: list[str]      # 树号，如 ["C15.378.553"]
    entry_terms: list[str]       # 入口词（同义词）
    subheadings: list[str]       # 允许的子树限定符


@dataclass
class ConceptDimension:
    """LLM 拆解出的一个概念维度。"""

    name: str                            # 维度名称（中文）
    english_terms: list[str]             # 英文自由词
    mesh_terms: list[MeshTerm]           # 验证后的 MeSH 术语（空=全部失败）
    failed_mesh: list[str]               # 验证失败的候选词
    synonyms: list[str]                  # 同义词/缩写


@dataclass
class SearchStrategy:
    """检索策略（双轨：Boolean + 语义）。"""

    boolean_query: str                   # Tier1 用的 PubMed Boolean 检索式
    semantic_query: str                  # Tier2 用的英文语义查询文本
    query_level: str = "medium"          # narrow / medium / broad
    concepts: list[ConceptDimension] = field(default_factory=list)
    filters: dict = field(default_factory=dict)
    total_count: int = 0                 # esearch 返回的总结果数
    fallback: bool = False               # True=LLM 拆解失败，用原始关键词


@dataclass
class Article:
    """文献（含检索元信息与语义评分）。"""

    pmid: str
    title: str
    abstract: str
    authors: str
    journal: str
    pub_date: str
    doi: str
    # 检索元信息
    source: str = "core"                 # core / related
    pubmed_rank: int = 0                 # PubMed 原始排序位置（0=第一篇）
    citation_count: int = 0
    # 语义评分
    semantic_score: float = 0.0          # MedCPT cosine 相似度
    rerank_score: float = 0.0            # Cross-Encoder 精排分数
    final_score: float = 0.0             # 综合评分
    # 标记
    has_abstract: bool = True
    embedding_cached: bool = False

    def to_dict(self) -> dict:
        """转换为字典（兼容现有 app.py 的文献字典格式）。"""
        return {
            "pmid": self.pmid,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "journal": self.journal,
            "pub_date": self.pub_date,
            "doi": self.doi,
        }
