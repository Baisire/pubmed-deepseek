"""查询理解 - LLM 概念拆解 + MeSH 验证 + 双轨查询生成。

职责：
- 用 DeepSeek 将自然语言查询拆解为结构化概念维度
- 并行验证各概念的 MeSH 候选词
- 生成三档 Boolean 检索式（窄/中/宽）+ 英文语义查询文本

双轨输出：
- boolean_query: 给 Tier1 PubMed esearch 用（MeSH + 自由词 + Boolean 逻辑）
- semantic_query: 给 Tier2 MedCPT 编码用（英文自然语言，表达研究意图）
"""

import json
import re
from typing import Optional

from openai import OpenAI

from .models import ConceptDimension, SearchStrategy
from .mesh_lookup import verify_mesh_batch

_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
_DEEPSEEK_MODEL = "deepseek-chat"

_CONCEPT_PROMPT = """你是一名医学文献检索专家，精通 PubMed MeSH 主题词体系和 Boolean 检索策略。

将用户的自然语言查询拆解为核心概念维度，每个维度提取英文检索词、MeSH候选词和同义词。

【输出格式】（严格 JSON，不要 Markdown 代码块标记）
{
  "core_topic": "一句话描述研究方向（英文）",
  "concepts": [
    {
      "name": "概念维度名称（中文）",
      "english_terms": ["英文检索词1", "英文检索词2"],
      "mesh_candidates": ["MeSH候选词1"],
      "synonyms": ["同义词1", "缩写1"]
    }
  ]
}

【规则】
1. 识别 2-4 个核心概念维度
2. mesh_candidates 必须是你确信存在的 MeSH 术语（如 Neutropenia, Monocytes）
3. english_terms 是标题/摘要中可能出现的自由词
4. synonyms 包括缩写、变体拼写、相关表述
5. 输出纯 JSON

【示例】
输入："化疗后中性粒细胞减少的预测因子"
输出：
{
  "core_topic": "hematological predictors for chemotherapy-induced febrile neutropenia",
  "concepts": [
    {
      "name": "中性粒细胞减少症",
      "english_terms": ["neutropenia", "febrile neutropenia"],
      "mesh_candidates": ["Neutropenia"],
      "synonyms": ["CIN", "FN", "neutrophil nadir"]
    },
    {
      "name": "血液学预测因子",
      "english_terms": ["monocyte", "lymphocyte", "monocytopenia", "lymphopenia"],
      "mesh_candidates": ["Monocytes", "Lymphocytes", "Lymphopenia"],
      "synonyms": ["AMC", "ALC", "monocyte count", "lymphocyte count"]
    },
    {
      "name": "化疗与预测",
      "english_terms": ["chemotherapy", "chemotherapy-induced", "predictor"],
      "mesh_candidates": ["Antineoplastic Agents"],
      "synonyms": ["cytotoxic chemotherapy", "risk factor", "prognostic"]
    }
  ]
}"""


def _parse_llm_json(raw: str) -> Optional[dict]:
    """容错解析 LLM 输出的 JSON。"""
    # 1. 剥离 Markdown 代码围栏
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()
    # 2. 尝试直接解析
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # 3. 正则提取首个 JSON 对象
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def _build_boolean_query(concepts: list[ConceptDimension], level: str) -> str:
    """根据概念维度和档位生成 Boolean 检索式。

    - narrow: 全 MeSH + 全 AND
    - medium: MeSH + 自由词 + AND（推荐默认）
    - broad: 自由词 + 同义词 + OR 为主
    """
    groups: list[str] = []

    for concept in concepts:
        mesh_parts = [f'"{m.term}"[MeSH Terms]' for m in concept.mesh_terms]
        free_parts = [f'"{t}"[Title/Abstract]' for t in concept.english_terms]

        if level == "narrow":
            # 全 MeSH + AND
            if mesh_parts:
                groups.append(f"({' OR '.join(mesh_parts)})")
            elif free_parts:
                groups.append(f"({' OR '.join(free_parts)})")
        elif level == "medium":
            # MeSH + 自由词 + AND
            all_parts = mesh_parts + free_parts
            if all_parts:
                groups.append(f"({' OR '.join(all_parts)})")
        else:  # broad
            # 自由词 + 同义词 + OR
            all_terms = concept.english_terms + concept.synonyms
            if all_terms:
                quoted = [f'"{t}"[Title/Abstract]' for t in all_terms]
                groups.append(f"({' OR '.join(quoted)})")

    return " AND ".join(groups) if groups else ""


def _build_semantic_query(core_topic: str, concepts: list[ConceptDimension]) -> str:
    """生成英文语义查询文本（供 MedCPT 编码用）。"""
    parts = [core_topic]
    for c in concepts:
        if c.english_terms:
            parts.append(f"{c.name}: {' '.join(c.english_terms[:3])}")
    return ". ".join(parts)


def understand_query(user_input: str, api_key: str,
                     filters: dict | None = None) -> SearchStrategy:
    """完整的查询理解流程：LLM 拆解 -> MeSH 验证 -> 双轨查询生成。

    Args:
        user_input: 用户的自然语言查询（中文或英文）
        api_key: DeepSeek API Key
        filters: 过滤条件 {date_range, article_types, language}

    Returns:
        SearchStrategy（含 boolean_query 和 semantic_query）
    """
    filters = filters or {}

    # 1. LLM 概念拆解
    try:
        client = OpenAI(api_key=api_key, base_url=_DEEPSEEK_BASE_URL)
        response = client.chat.completions.create(
            model=_DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": _CONCEPT_PROMPT},
                {"role": "user", "content": user_input},
            ],
            temperature=0.1,
            max_tokens=2000,
        )
        raw_output = response.choices[0].message.content
        parsed = _parse_llm_json(raw_output)
    except Exception:
        parsed = None

    # 降级：LLM 失败，用原始关键词
    if not parsed or "concepts" not in parsed:
        return SearchStrategy(
            boolean_query=user_input,
            semantic_query=user_input,
            query_level="medium",
            fallback=True,
            filters=filters,
        )

    core_topic = parsed.get("core_topic", user_input)

    # 2. MeSH 验证（并行）
    all_candidates: list[str] = []
    for c in parsed["concepts"]:
        all_candidates.extend(c.get("mesh_candidates", []))
    all_candidates = list(set(all_candidates))  # 去重

    mesh_results = verify_mesh_batch(all_candidates) if all_candidates else {}

    # 3. 构建 ConceptDimension 列表
    concepts: list[ConceptDimension] = []
    for c in parsed["concepts"]:
        mesh_terms = []
        failed = []
        for candidate in c.get("mesh_candidates", []):
            result = mesh_results.get(candidate)
            if result is not None:
                mesh_terms.append(result)
            else:
                failed.append(candidate)

        concepts.append(ConceptDimension(
            name=c.get("name", ""),
            english_terms=c.get("english_terms", []),
            mesh_terms=mesh_terms,
            failed_mesh=failed,
            synonyms=c.get("synonyms", []),
        ))

    # 4. 生成双轨查询
    medium_query = _build_boolean_query(concepts, "medium")
    narrow_query = _build_boolean_query(concepts, "narrow")
    broad_query = _build_boolean_query(concepts, "broad")
    semantic_query = _build_semantic_query(core_topic, concepts)

    # 默认用中档
    boolean_query = medium_query if medium_query else user_input

    return SearchStrategy(
        boolean_query=boolean_query,
        semantic_query=semantic_query if semantic_query else user_input,
        query_level="medium",
        concepts=concepts,
        filters=filters,
    )
