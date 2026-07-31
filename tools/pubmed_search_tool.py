"""PubMed 检索工具 - 调用 v3.0 混合语义检索管道。"""

from typing import Any

from search.pipeline import search as v3_search


TOOL_NAME = "pubmed_search"

TOOL_DESCRIPTION = (
    "对 PubMed 医学文献数据库执行混合语义检索（v3.0）。"
    "结合 MeSH 术语解析、结构化召回、语义精排与引文增强，返回高相关度文献列表。"
    "支持中英文自然语言查询，自动转换为 PubMed Boolean 检索式。"
)

TOOL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": (
                "用户的检索查询字符串，支持中英文自然语言。"
                "例如：'type 2 diabetes metformin treatment' 或 '2型糖尿病二甲双胍治疗'"
            ),
        },
        "max_results": {
            "type": "integer",
            "description": "返回的文献数量上限，默认 10，建议范围 5-20",
            "default": 10,
            "minimum": 1,
            "maximum": 50,
        },
        "use_cross_encoder": {
            "type": "boolean",
            "description": "是否启用 Cross-Encoder 精排以提升排序精度，默认 True。关闭可降低延迟。",
            "default": True,
        },
        "use_citations": {
            "type": "boolean",
            "description": "是否获取引文数据并用于综合评分，默认 True。",
            "default": True,
        },
    },
    "required": ["query"],
    "additionalProperties": False,
}


def execute(query: str,
            max_results: int = 10,
            use_cross_encoder: bool = True,
            use_citations: bool = True,
            context: dict[str, Any] | None = None) -> dict[str, Any]:
    """执行 PubMed 混合语义检索。

    Args:
        query: 用户检索查询
        max_results: 返回文献数
        use_cross_encoder: 是否启用 Cross-Encoder 精排
        use_citations: 是否启用引文增强
        context: 调用上下文，含 deepseek_api_key、ncbi_api_key 等

    Returns:
        包含 articles、total_count、quality_assessment 的结果字典；
        失败时返回 {"error": "原因"}
    """
    context = context or {}
    deepseek_api_key = context.get("deepseek_api_key", "")
    ncbi_api_key = context.get("ncbi_api_key", "")

    if not deepseek_api_key:
        return {"error": "缺少 DeepSeek API Key，无法进行查询理解和语义检索"}

    if not query or not query.strip():
        return {"error": "查询内容不能为空"}

    try:
        result = v3_search(
            user_input=query.strip(),
            deepseek_api_key=deepseek_api_key,
            ncbi_api_key=ncbi_api_key,
            max_results=max_results,
            candidate_pool_size=max(50, max_results * 3),
            expand_related=False,
            use_cross_encoder=use_cross_encoder,
            use_citations=use_citations,
        )
    except Exception as e:
        return {"error": f"PubMed 检索失败：{e}"}

    articles_out: list[dict[str, Any]] = []
    for article in result.articles:
        articles_out.append({
            "pmid": article.pmid,
            "title": article.title,
            "abstract": article.abstract,
            "authors": article.authors,
            "journal": article.journal,
            "pub_date": article.pub_date,
            "doi": article.doi,
            "semantic_score": round(article.semantic_score, 4),
            "final_score": round(article.final_score, 4),
        })

    total_count = result.strategy.total_count if result.strategy else len(result.articles)

    return {
        "articles": articles_out,
        "total_count": total_count,
        "quality_assessment": result.quality,
    }
