"""文献分析工具 - 基于 DeepSeek 对指定 PMID 列表的文献做翻译/摘要/综述。"""

import json
from typing import Any

import requests
from openai import OpenAI

from search.pubmed_recall import _efetch


TOOL_NAME = "analyze_literature"

TOOL_DESCRIPTION = (
    "对指定 PMID 列表的 PubMed 文献进行 AI 分析（翻译、摘要或综述），"
    "调用 DeepSeek 大模型生成 Markdown 格式的结构化报告。"
    "三种模式：translate（翻译）、summary（摘要）、review（综述）。"
)

TOOL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "pmids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "需要分析的文献 PMID 列表，例如 ['38123456', '38098765']",
        },
        "analysis_type": {
            "type": "string",
            "enum": ["translate", "summary", "review"],
            "description": (
                "分析类型：translate=中文翻译保留术语对照，"
                "summary=提炼核心观点，review=跨文献对比综述"
            ),
        },
        "keyword": {
            "type": "string",
            "description": "检索关键词，在 review（综述）模式下用于报告标题和聚焦分析，其他模式可选",
            "default": "",
        },
    },
    "required": ["pmids", "analysis_type"],
    "additionalProperties": False,
}


# ============================================================================
# System Prompt（与 app.py / Coze 方案一致）
# ============================================================================

SYSTEM_PROMPT_TRANSLATE = """你是一位专业的医学文献翻译专家，精通英中双语医学翻译。

你的任务是将 PubMed 检索到的英文医学文献翻译为中文，要求准确、专业、可读性强。

==翻译规则==
1. 医学术语保留英文原文对照，格式为：中文翻译（English Term）
   例：心肌梗死（myocardial infarction）、经皮冠状动脉介入治疗（percutaneous coronary intervention）
2. 药物名称保留通用名：如 阿司匹林（aspirin）、氯吡格雷（clopidogrel）
3. 疾病名称使用中文医学标准译名
4. 实验方法名称保留对照：如 随机对照试验（randomized controlled trial, RCT）
5. 统计学术语保留对照：如 相对风险（relative risk, RR）、置信区间（confidence interval, CI）
6. 保持原文逻辑结构，不增减信息
7. 译不出或不确定的术语保留英文原文并标注 [待确认]

==输出格式==
对每篇文献输出以下结构：

### 文献 {序号}
- **PMID**：{PMID}
- **英文标题**：{原文标题}
- **中文标题**：{翻译标题}
- **作者**：{作者列表}
- **期刊**：{期刊名称}
- **发表日期**：{日期}

#### 摘要翻译
{中文翻译摘要，分段对应原文结构}

#### 关键术语对照
| 英文 | 中文 |
|------|------|
| {term} | {translation} |

---

==限制条件==
1. 不得编造文献信息
2. 不得遗漏摘要中的任何段落
3. 不得自行添加原文没有的结论
4. 检索到 {count} 篇文献，必须全部翻译"""

SYSTEM_PROMPT_SUMMARY = """你是一位医学文献分析专家，擅长快速提炼学术论文的核心内容。

你的任务是对 PubMed 检索到的医学文献进行摘要总结，提炼每篇文献的核心观点。

==分析要求==
对每篇文献提取以下信息：
1. 研究目的：本研究试图解决什么问题
2. 研究方法：研究设计、样本量、干预措施、随访时间
3. 主要发现：核心结果，包含量化数据（如有）
4. 研究结论：作者得出的主要结论
5. 局限性：研究存在的不足（如原文提及）
6. 临床意义：对临床实践的指导价值

==输出格式==
对每篇文献输出以下结构：

### 文献 {序号}：{中文标题}

**基本信息**
- PMID：{PMID}
- 期刊：{期刊}
- 发表日期：{日期}

**研究目的**
{1-2 句话概括}

**研究方法**
{研究设计、样本、干预等}

**主要发现**
- {发现1，含关键数据}
- {发现2}
- {发现3}

**研究结论**
{1-2 句话概括}

**局限性**
{如有则列出，无则标注"原文未提及"}

**临床意义**
{对临床实践的指导价值}

---

==限制条件==
1. 总结必须基于原文摘要内容，不得编造
2. 量化数据必须引用原文数字，不得四舍五入或估算
3. 每篇文献的总结控制在 200-400 字
4. 检索到 {count} 篇文献，必须全部总结
5. 如果某篇文献无摘要，标注"该文献暂无摘要"
"""

SYSTEM_PROMPT_REVIEW = """你是一位医学文献综述写作专家，擅长跨文献对比分析和综述撰写。

你的任务是基于 PubMed 检索到的多篇医学文献，撰写一份结构化的文献综述报告。

==综述结构==
报告必须包含以下部分：

# 文献综述：{关键词}

## 一、检索概览
- 检索数据库：PubMed
- 检索关键词：{keyword}
- 检索结果：共 {count} 篇文献
- 文献列表（表格形式）：
  | 序号 | PMID | 标题 | 期刊 | 日期 |
  |------|------|------|------|------|

## 二、研究现状概述
{300-500 字概述该领域的研究现状，基于检索到的文献}

## 三、文献逐篇分析
对每篇文献简要分析（100-200字/篇）：
### 文献 {序号}
- **标题**：{中文标题}
- **核心发现**：{关键结论}
- **研究方法**：{方法简述}

## 四、对比分析
### 4.1 研究方法对比
| 文献 | 研究设计 | 样本量 | 干预/暴露 | 主要终点 |
|------|---------|--------|----------|---------|
{逐篇对比}

### 4.2 结论一致性分析
{分析各文献结论是否一致，存在哪些共识和分歧}

### 4.3 量化结果对比
{对比关键数据，如疗效指标、不良反应发生率等}

## 五、研究趋势与空白
### 5.1 研究趋势
{基于检索文献总结的研究方向趋势}

### 5.2 研究空白
{当前研究的不足和未来方向}

## 六、总结
{200-300 字总结，概括检索文献的整体发现和对临床/研究的指导意义}

## 参考文献
{按序号列出所有文献的引用格式：作者. 标题. 期刊. 日期. PMID. DOI.}

---

==写作要求==
1. 综述必须基于检索到的文献，不得编造或引用未检索的文献
2. 对比分析必须客观，不偏袒任何一篇文献的结论
3. 量化数据必须引用原文数字
4. 研究趋势和空白基于检索文献的实际内容推导，不得过度推测
5. 参考文献格式统一
6. 整体报告控制在 2000-4000 字"""

MODE_CONFIG: dict[str, dict[str, Any]] = {
    "translate": {"system_prompt": SYSTEM_PROMPT_TRANSLATE, "temperature": 0.3},
    "summary": {"system_prompt": SYSTEM_PROMPT_SUMMARY, "temperature": 0.4},
    "review": {"system_prompt": SYSTEM_PROMPT_REVIEW, "temperature": 0.5},
}

MAX_TOKENS = 8192

_USER_PROMPT_TEMPLATE = (
    "检索关键词：{keyword}\n"
    "分析模式：{analysis_type}\n"
    "文献数量：{count}\n\n"
    "文献数据（JSON格式）：\n{articles_json}\n\n"
    "请根据分析模式对以上文献进行处理，输出Markdown格式的报告。"
)


def _fetch_articles(pmids: list[str], ncbi_api_key: str = "") -> list[dict[str, Any]]:
    """根据 PMID 列表从 PubMed 获取文献详情。"""
    try:
        article_objs = _efetch(pmids, api_key=ncbi_api_key)
    except requests.RequestException as e:
        raise RuntimeError(f"PubMed efetch 失败：{e}") from e

    return [a.to_dict() for a in article_objs]


def _call_deepseek(api_key: str, system_prompt: str,
                   user_prompt: str, temperature: float) -> str:
    """调用 DeepSeek API 生成分析报告。"""
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=MAX_TOKENS,
    )
    content = response.choices[0].message.content or ""
    if not content.strip():
        raise RuntimeError("DeepSeek 返回了空内容")
    return content


def execute(pmids: list[str],
            analysis_type: str,
            keyword: str = "",
            context: dict[str, Any] | None = None) -> dict[str, Any]:
    """对指定 PMID 列表的文献执行 DeepSeek 分析。

    Args:
        pmids: PMID 列表
        analysis_type: 分析类型 translate/summary/review
        keyword: 检索关键词（review 模式必填）
        context: 调用上下文，含 deepseek_api_key、ncbi_api_key 等

    Returns:
        包含 report（Markdown 字符串）的结果字典；
        失败时返回 {"error": "原因"}
    """
    context = context or {}
    deepseek_api_key = context.get("deepseek_api_key", "")
    ncbi_api_key = context.get("ncbi_api_key", "")

    if not deepseek_api_key:
        return {"error": "缺少 DeepSeek API Key，无法进行文献分析"}

    if not pmids:
        return {"error": "pmids 列表不能为空"}

    # 过滤空字符串并去重保序
    seen: set[str] = set()
    clean_pmids: list[str] = []
    for pid in pmids:
        p = pid.strip() if isinstance(pid, str) else ""
        if p and p not in seen:
            seen.add(p)
            clean_pmids.append(p)

    if not clean_pmids:
        return {"error": "pmids 列表中没有有效 PMID"}

    if analysis_type not in MODE_CONFIG:
        return {
            "error": (
                f"无效的 analysis_type：{analysis_type}，"
                f"可选值：{', '.join(MODE_CONFIG.keys())}"
            )
        }

    if analysis_type == "review" and not keyword:
        return {"error": "review 模式必须提供 keyword 参数"}

    # 1. 获取文献详情
    try:
        articles = _fetch_articles(clean_pmids, ncbi_api_key)
    except Exception as e:
        return {"error": f"获取文献详情失败：{e}"}

    if not articles:
        return {"error": "未获取到任何文献，请检查 PMID 是否正确"}

    # 2. 构建提示词并调用 DeepSeek
    config = MODE_CONFIG[analysis_type]
    count = len(articles)
    system_prompt = config["system_prompt"].replace("{count}", str(count))
    if analysis_type == "review":
        system_prompt = system_prompt.replace("{keyword}", keyword)

    articles_json = json.dumps(articles, ensure_ascii=False, indent=2)
    user_prompt = _USER_PROMPT_TEMPLATE.format(
        keyword=keyword or "(未提供)",
        analysis_type=analysis_type,
        count=count,
        articles_json=articles_json,
    )

    try:
        report = _call_deepseek(
            api_key=deepseek_api_key,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=config["temperature"],
        )
    except Exception as e:
        return {"error": f"DeepSeek API 调用失败：{e}"}

    return {
        "report": report,
        "article_count": count,
        "analysis_type": analysis_type,
    }
