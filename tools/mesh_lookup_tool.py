"""MeSH 术语查询工具 - 验证候选词并获取 MeSH 详细信息。"""

from typing import Any

from search.mesh_lookup import verify_mesh_batch


TOOL_NAME = "lookup_mesh"

TOOL_DESCRIPTION = (
    "查询候选术语是否为有效的 MeSH（医学主题词表）术语，"
    "返回其 D 编号、树号、入口词等详细信息。"
    "用于在检索前确认 MeSH 术语准确性，或扩展检索关键词。"
)

TOOL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "terms": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "待验证的候选 MeSH 术语列表，建议使用英文术语。"
                "例如：['diabetes mellitus', 'metformin', 'hypertension']"
            ),
        },
    },
    "required": ["terms"],
    "additionalProperties": False,
}


def execute(terms: list[str],
            context: dict[str, Any] | None = None) -> dict[str, Any]:
    """批量验证 MeSH 候选术语。

    Args:
        terms: 候选 MeSH 术语列表
        context: 调用上下文（此工具暂不需要）

    Returns:
        包含 valid_terms 和 failed_terms 的结果字典；
        失败时返回 {"error": "原因"}
    """
    if not terms:
        return {"error": "terms 列表不能为空"}

    # 过滤空字符串并去重保序
    seen: set[str] = set()
    clean_terms: list[str] = []
    for t in terms:
        t_stripped = t.strip() if isinstance(t, str) else ""
        if t_stripped and t_stripped not in seen:
            seen.add(t_stripped)
            clean_terms.append(t_stripped)

    if not clean_terms:
        return {"error": "terms 列表中没有有效术语"}

    try:
        results = verify_mesh_batch(clean_terms)
    except Exception as e:
        return {"error": f"MeSH 查询失败：{e}"}

    valid_terms: list[dict[str, Any]] = []
    failed_terms: list[str] = []

    for term in clean_terms:
        mesh = results.get(term)
        if mesh is None:
            failed_terms.append(term)
            continue
        valid_terms.append({
            "term": mesh.term,
            "ui": mesh.ui,
            "tree_numbers": mesh.tree_numbers,
            "entry_terms": mesh.entry_terms,
            "subheadings": mesh.subheadings,
        })

    return {
        "valid_terms": valid_terms,
        "failed_terms": failed_terms,
    }
