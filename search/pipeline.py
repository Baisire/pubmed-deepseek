"""检索管道编排 - 串联查询理解 -> Tier1 召回 -> Tier2 精排 -> 引文增强。

这是 app.py 调用检索功能的统一入口，封装完整的 v3.0 两段式混合语义检索流程。
"""

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from .models import Article, SearchStrategy
from .query_understanding import understand_query
from .pubmed_recall import recall as pubmed_recall
from .semantic_rerank import SemanticReranker
from .citation_boost import score_and_rank


@dataclass
class SearchResult:
    """完整检索结果。"""

    articles: list[Article] = field(default_factory=list)
    strategy: Optional[SearchStrategy] = None
    timing: dict = field(default_factory=dict)
    quality: dict = field(default_factory=dict)


def search(user_input: str,
           deepseek_api_key: str,
           ncbi_api_key: str = "",
           max_results: int = 10,
           candidate_pool_size: int = 50,
           expand_related: bool = False,
           use_cross_encoder: bool = True,
           use_citations: bool = True,
           progress_callback: Optional[Callable[[str], None]] = None) -> SearchResult:
    """v3.0 完整检索管道。

    Args:
        user_input: 用户自然语言查询
        deepseek_api_key: DeepSeek API Key（用于查询理解）
        ncbi_api_key: NCBI API Key（可选，提升频率限制）
        max_results: 最终返回的文献数
        candidate_pool_size: Tier1 候选池大小
        expand_related: 是否启用 elink 相关文献扩展
        use_cross_encoder: 是否启用 Cross-Encoder 精排（关闭可降低延迟）
        use_citations: 是否获取引文数据
        progress_callback: 进度回调函数

    Returns:
        SearchResult（含排序后的文献列表、检索策略、计时、质量指标）
    """
    result = SearchResult()
    timer: dict[str, float] = {}

    def _log(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    # Step 0: 查询理解
    _log("正在理解查询意图...")
    t0 = time.time()
    strategy = understand_query(user_input, deepseek_api_key)
    timer["query_understanding"] = time.time() - t0
    result.strategy = strategy

    if strategy.fallback:
        _log("智能优化不可用，使用基础检索")
    else:
        concept_count = len(strategy.concepts)
        _log(f"识别 {concept_count} 个概念维度")

    # Step 1: Tier 1 结构化召回
    _log("正在检索 PubMed...")
    t0 = time.time()
    articles = pubmed_recall(
        strategy,
        api_key=ncbi_api_key,
        max_results=candidate_pool_size,
        expand_related=expand_related,
    )
    timer["tier1_recall"] = time.time() - t0
    _log(f"召回 {len(articles)} 篇候选文献")

    if not articles:
        _log("未检索到文献")
        result.timing = timer
        return result

    # Step 2: Tier 2 语义精排
    _log("正在语义精排...")
    t0 = time.time()
    reranker = SemanticReranker()
    articles = reranker.rerank_articles(
        strategy.semantic_query,
        articles,
        use_cross_encoder=use_cross_encoder,
    )
    timer["tier2_rerank"] = time.time() - t0

    if use_cross_encoder:
        _log("语义精排完成（含 Cross-Encoder 重排序）")
    else:
        _log("语义精排完成（仅 bi-encoder）")

    # Step 3: 引文增强与综合评分
    _log("正在计算综合评分...")
    t0 = time.time()
    articles = score_and_rank(articles, api_key=ncbi_api_key,
                              use_citations=use_citations)
    timer["citation_scoring"] = time.time() - t0

    # 截取 Top N
    result.articles = articles[:max_results]
    result.timing = timer

    # 质量评估
    result.quality = _evaluate_quality(result.articles, strategy)
    _log(f"检索完成：{len(result.articles)} 篇，质量评估：{result.quality.get('assessment', 'N/A')}")

    return result


def _evaluate_quality(articles: list[Article],
                      strategy: SearchStrategy) -> dict:
    """检索质量自评。"""
    if not articles:
        return {"assessment": "poor", "suggestions": ["未检索到文献"]}

    total = len(articles)
    scores = [a.final_score for a in articles]
    avg_score = sum(scores) / len(scores) if scores else 0
    top_score = max(scores) if scores else 0

    # 评分分布区分度
    if len(scores) >= 4:
        sorted_scores = sorted(scores, reverse=True)
        top_mean = sum(sorted_scores[:len(sorted_scores)//2]) / (len(sorted_scores)//2)
        bottom_mean = sum(sorted_scores[len(sorted_scores)//2:]) / (len(sorted_scores) - len(sorted_scores)//2)
        discrimination = top_mean - bottom_mean
    else:
        discrimination = 0.0

    has_abstract_count = sum(1 for a in articles if a.has_abstract)
    abstract_coverage = has_abstract_count / total if total > 0 else 0

    # 评估等级
    if total >= 5 and avg_score > 0.5 and discrimination > 0.1:
        assessment = "good"
    elif total >= 3:
        assessment = "fair"
    else:
        assessment = "poor"

    suggestions: list[str] = []
    if abstract_coverage < 0.7:
        suggestions.append("部分文献缺少摘要，语义匹配精度可能受影响")
    if total < 5:
        suggestions.append("检索结果较少，建议尝试扩大检索范围")
    if discrimination < 0.05:
        suggestions.append("文献区分度低，结果排序可能不够精准")

    return {
        "total": total,
        "avg_score": round(avg_score, 4),
        "top_score": round(top_score, 4),
        "discrimination": round(discrimination, 4),
        "abstract_coverage": round(abstract_coverage, 4),
        "assessment": assessment,
        "suggestions": suggestions,
    }
