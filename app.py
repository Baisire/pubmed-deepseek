"""医学 PubMed 检索 + DeepSeek 文献解读 - Streamlit Web 应用

基于 Coze 工作流方案的轻量化替代实现，功能完全一致：
- PubMed E-utilities API 检索文献（esearch + efetch）
- DeepSeek 大模型三模式分析（翻译 / 摘要 / 综述）
- Markdown 格式报告展示与下载

运行方式：
    pip install -r requirements.txt
    set DEEPSEEK_API_KEY=sk-xxxx
    streamlit run app.py
"""

import json
import os
import xml.etree.ElementTree as ET
from typing import Any, Optional

import requests
import streamlit as st
from openai import OpenAI

import db
import tier_gating

# chat / llm 模块（对话模式使用）
from chat import render_sidebar_sessions, render_chat_area, get_or_create_session, append_message
from llm import (
    get_available_models,
    get_model_config,
    get_provider_for_model,
    resolve_api_key,
    chat_completion,
    LLMResponse,
    LLMError,
)

# tools 模块可选导入，缺失时对话模式降级为纯文本
try:
    from tools.tool_registry import TOOL_REGISTRY  # type: ignore
except Exception:
    TOOL_REGISTRY: Optional[dict] = None

# ============================================================================
# 环境变量（必须在 import search 前设置，避免 C 盘生成文件）
# ============================================================================

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(os.path.join(_DATA_DIR, "hf_cache"), exist_ok=True)
os.environ.setdefault("HF_HOME", os.path.join(_DATA_DIR, "hf_cache"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from search.pipeline import search as v3_search, SearchResult  # noqa: E402

# ============================================================================
# 常量定义
# ============================================================================

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

# 三套 System Prompt（从 docs/提示词设计/ 完整复用，与 Coze 方案保持一致）
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

# 模式配置：system prompt + temperature（与 Coze 方案一致）
MODE_CONFIG = {
    "translate": {"system_prompt": SYSTEM_PROMPT_TRANSLATE, "temperature": 0.3},
    "summary": {"system_prompt": SYSTEM_PROMPT_SUMMARY, "temperature": 0.4},
    "review": {"system_prompt": SYSTEM_PROMPT_REVIEW, "temperature": 0.5},
}

MAX_TOKENS = 8192

# ============================================================================
# 界面国际化文案
# ============================================================================

STRINGS = {
    "zh": {
        "page_title": "PubMed 文献检索 + DeepSeek 解读",
        "title": "📚 医学 PubMed 检索 + DeepSeek 文献解读",
        "caption": "输入医学关键词，自动检索 PubMed 文献，由 DeepSeek AI 完成翻译/摘要/综述",
        "lang_label": "界面语言",
        "sidebar_header": "检索参数",
        "keyword_label": "关键词",
        "keyword_help": "建议使用英文关键词，可使用 MeSH 主题词提高检索精度",
        "max_results_label": "文献数量",
        "analysis_label": "分析模式",
        "analysis_translate": "翻译 - 中文翻译，保留术语对照",
        "analysis_summary": "摘要 - 提炼核心观点",
        "analysis_review": "综述 - 跨文献对比分析",
        "api_key_label": "DeepSeek API Key",
        "api_key_help": "已通过环境变量 DEEPSEEK_API_KEY 配置则无需重复填写",
        "run_btn": "开始检索",
        "cost_note": "DeepSeek 按 token 计费（约 0.01 元/次）",
        "warn_empty_keyword": "请输入检索关键词",
        "error_no_api_key": "缺少 DeepSeek API Key。请在侧边栏填写，或设置环境变量：set DEEPSEEK_API_KEY=sk-xxxx",
        "spinner_searching": "正在检索 PubMed...",
        "error_no_results": "未检索到相关文献，请尝试更换关键词或扩大检索范围。",
        "info_keyword_tip": "提示：使用英文关键词效果更佳，如 'diabetes treatment' 而非 '糖尿病治疗'",
        "success_found": "检索到 {count} 篇文献",
        "article_list_header": "文献列表",
        "view_abstract": "查看英文原文摘要",
        "spinner_analyzing": "DeepSeek 正在分析文献...",
        "error_api_failed": "DeepSeek API 调用失败：{error}",
        "info_retry": "请检查 API Key 是否正确，或稍后重试。",
        "error_empty_response": "DeepSeek 返回了空内容，请稍后重试。",
        "download_btn": "下载报告 (Markdown)",
        "footer": "点击文献标题可跳转 PubMed 查看原文页面 | DOI 可在 https://doi.org/ 查询全文",
        "user_prompt": "检索关键词：{keyword}\n分析模式：{analysis_type}\n文献数量：{count}\n\n文献数据（JSON格式）：\n{articles_json}\n\n请根据分析模式对以上文献进行处理，输出Markdown格式的报告。",
        "login_title": "登录",
        "login_username": "用户名",
        "login_password": "密码",
        "login_btn": "登录",
        "login_failed": "用户名或密码错误，或账号已被禁用",
        "login_hint": "如无账号，请联系管理员分配。",
        "logout_btn": "登出",
        "welcome": "欢迎，{username}",
        "admin_panel_title": "管理后台",
        "admin_add_user": "添加客户",
        "admin_new_username": "新用户名",
        "admin_new_password": "新密码",
        "admin_create_btn": "创建",
        "admin_user_list": "客户列表",
        "admin_usage_stats": "用量统计",
        "admin_recent_usage": "最近检索",
        "admin_delete": "删除",
        "admin_reset_pwd": "重置密码",
        "admin_new_password_prompt": "输入新密码",
        "admin_no_delete_self": "不能删除自己",
        "admin_no_delete_admin": "不能删除管理员账号",
        "my_history": "我的检索历史",
        "no_history": "暂无检索记录",
        "api_key_input_help": "请输入您的 DeepSeek API Key",
        "admin_admin_role": "管理员",
        "admin_user_role": "客户",
        "admin_active": "启用",
        "admin_disabled": "禁用",
        "admin_role": "角色",
        "admin_status": "状态",
        "admin_total_searches": "检索次数",
        "admin_total_articles": "文献总数",
        "admin_last_search": "最后检索",
        "admin_search_test": "检索测试",
        "history_time": "检索时间",
        # ---- v3.0 混合语义检索 ----
        "semantic_header": "语义检索设置",
        "semantic_rerank_label": "启用语义精排",
        "semantic_rerank_help": "使用 MedCPT 生物医学向量模型做语义重排序，提升相关性",
        "cross_encoder_label": "启用 Cross-Encoder",
        "cross_encoder_help": "使用 bge-reranker 深度精排，更精准但速度较慢",
        "citation_label": "启用引文排序",
        "citation_help": "结合被引数做综合评分，高被引文献优先",
        "candidate_pool_label": "候选池大小",
        "candidate_pool_help": "从 PubMed 召回的候选文献数，越大越全但越慢",
        "progress_title": "检索进度",
        "strategy_expander": "检索式展开（可查看 Boolean / Semantic 检索式）",
        "strategy_boolean": "Boolean 检索式（PubMed 召回）",
        "strategy_semantic": "Semantic 检索式（语义精排）",
        "strategy_fallback": "降级模式（关键词原样传入）",
        "article_cards_header": "检索结果（按综合评分排序）",
        "semantic_score_label": "语义分数",
        "citation_count_label": "被引数",
        "pubmed_rank_label": "PubMed 原始排名",
        "source_label": "来源",
        "source_core": "核心检索",
        "source_related": "相关文献扩展",
        "quality_panel_title": "检索质量自评",
        "quality_total": "结果总数",
        "quality_avg_score": "平均语义分",
        "quality_top_score": "最高语义分",
        "quality_discrimination": "区分度",
        "quality_abstract_coverage": "摘要覆盖率",
        "quality_assessment": "评估等级",
        "quality_good": "良好",
        "quality_fair": "一般",
        "quality_poor": "较差",
        "quality_suggestions": "优化建议",
        "search_v3_failed": "v3.0 检索失败，已降级为 v1.0 检索：{error}",
        # ---- M7 双模式 ----
        "mode_search": "🔍 检索模式",
        "mode_chat": "💬 对话模式",
        "chat_sidebar_header": "对话设置",
        "chat_model_label": "选择模型",
        "chat_api_key_label": "API Key 配置",
        "chat_provider_label": "{provider} API Key",
        "chat_provider_help": "填入对应提供商的 API Key，将加密保存在本地数据库",
        "chat_save_key_btn": "保存 Key",
        "chat_key_saved": "API Key 已保存",
        "chat_key_save_failed": "保存 API Key 失败：{error}",
        "chat_title": "💬 医学文献对话助手",
        "chat_caption": "基于多模型大语言模型的智能医学问答，支持 PubMed 检索工具调用",
        "chat_upgrade_tip": "💡 对话模式仅对 Basic 及以上用户开放，如需升级请联系管理员。",
        "chat_discuss_btn": "在对话中讨论这些文献",
        "chat_discuss_sent": "已跳转到对话模式，文献信息已自动发送。",
        "chat_init_failed": "对话模式初始化失败：{error}",
        "chat_no_api_key": "当前所选模型没有可用的 API Key，请在侧边栏配置。",
        # ---- 套餐订阅 ----
        "subscription_btn": "升级套餐",
        "subscription_title": "套餐与订阅",
        "subscription_subtitle": "选择适合您的方案，解锁更多检索能力",
        "subscription_current": "当前套餐",
        "subscription_contact_btn": "联系管理员升级",
        "subscription_contact_msg": "请联系管理员开通升级\n邮箱：admin@example.com\n微信：admin_wechat",
        "subscription_compare": "功能对比",
        "subscription_back": "返回检索",
        "subscription_daily_search": "每日检索次数",
        "subscription_max_articles": "每次文献数",
        "subscription_feature_semantic": "语义精排",
        "subscription_feature_cross_encoder": "Cross-Encoder 精排",
        "subscription_feature_citation": "引文排序",
        "subscription_feature_chat": "对话模式",
        "subscription_feature_tools": "工具调用",
        "subscription_feature_models": "可用模型",
    },
    "en": {
        "page_title": "PubMed Search + DeepSeek Analysis",
        "title": "📚 PubMed Search + DeepSeek Analysis",
        "caption": "Enter medical keywords to search PubMed, analyzed by DeepSeek AI for translation/summary/review",
        "lang_label": "Language",
        "sidebar_header": "Search Parameters",
        "keyword_label": "Keyword",
        "keyword_help": "English keywords recommended. MeSH terms improve search accuracy",
        "max_results_label": "Number of Articles",
        "analysis_label": "Analysis Mode",
        "analysis_translate": "Translate - Chinese translation with term mapping",
        "analysis_summary": "Summary - Extract key points",
        "analysis_review": "Review - Cross-article comparative analysis",
        "api_key_label": "DeepSeek API Key",
        "api_key_help": "Pre-configured via DEEPSEEK_API_KEY env var if set",
        "run_btn": "Search",
        "cost_note": "PubMed API is free. DeepSeek charges ~$0.01 per call",
        "warn_empty_keyword": "Please enter a search keyword",
        "error_no_api_key": "Missing DeepSeek API Key. Enter it in the sidebar, or set env var: set DEEPSEEK_API_KEY=sk-xxxx",
        "spinner_searching": "Searching PubMed...",
        "error_no_results": "No articles found. Try different keywords or broaden your search.",
        "info_keyword_tip": "Tip: English keywords work better, e.g. 'diabetes treatment' instead of '糖尿病治疗'",
        "success_found": "Found {count} articles",
        "article_list_header": "Article List",
        "view_abstract": "View Original Abstract",
        "spinner_analyzing": "DeepSeek is analyzing articles...",
        "error_api_failed": "DeepSeek API call failed: {error}",
        "info_retry": "Check your API Key or try again later.",
        "error_empty_response": "DeepSeek returned empty content. Please try again.",
        "download_btn": "Download Report (Markdown)",
        "footer": "Click article title to view on PubMed | DOI can be looked up at https://doi.org/",
        "user_prompt": "Search keyword: {keyword}\nAnalysis mode: {analysis_type}\nArticle count: {count}\n\nArticle data (JSON):\n{articles_json}\n\nPlease process the above articles according to the analysis mode and output a Markdown report.",
        "login_title": "Login",
        "login_username": "Username",
        "login_password": "Password",
        "login_btn": "Login",
        "login_failed": "Invalid username or password, or account disabled",
        "login_hint": "No account? Contact your administrator.",
        "logout_btn": "Logout",
        "welcome": "Welcome, {username}",
        "admin_panel_title": "Admin Panel",
        "admin_add_user": "Add Customer",
        "admin_new_username": "New Username",
        "admin_new_password": "New Password",
        "admin_create_btn": "Create",
        "admin_user_list": "Customer List",
        "admin_usage_stats": "Usage Statistics",
        "admin_recent_usage": "Recent Searches",
        "admin_delete": "Delete",
        "admin_reset_pwd": "Reset Password",
        "admin_new_password_prompt": "Enter new password",
        "admin_no_delete_self": "Cannot delete yourself",
        "admin_no_delete_admin": "Cannot delete admin account",
        "my_history": "My Search History",
        "no_history": "No search records yet",
        "api_key_input_help": "Please enter your DeepSeek API Key",
        "admin_admin_role": "Admin",
        "admin_user_role": "Customer",
        "admin_active": "Active",
        "admin_disabled": "Disabled",
        "admin_role": "Role",
        "admin_status": "Status",
        "admin_total_searches": "Searches",
        "admin_total_articles": "Articles",
        "admin_last_search": "Last Search",
        "admin_search_test": "Search Test",
        "history_time": "Time",
        # ---- v3.0 Hybrid Semantic Search ----
        "semantic_header": "Semantic Search Settings",
        "semantic_rerank_label": "Enable Semantic Rerank",
        "semantic_rerank_help": "Use MedCPT biomedical vector model for semantic re-ranking",
        "cross_encoder_label": "Enable Cross-Encoder",
        "cross_encoder_help": "Use bge-reranker for deep reranking (more accurate but slower)",
        "citation_label": "Enable Citation Boost",
        "citation_help": "Combine citation count for composite scoring, prioritizing highly cited papers",
        "candidate_pool_label": "Candidate Pool Size",
        "candidate_pool_help": "Number of candidate articles recalled from PubMed",
        "progress_title": "Search Progress",
        "strategy_expander": "Search Query Details (Boolean / Semantic)",
        "strategy_boolean": "Boolean Query (PubMed Recall)",
        "strategy_semantic": "Semantic Query (Reranking)",
        "strategy_fallback": "Fallback Mode (raw keyword)",
        "article_cards_header": "Results (sorted by composite score)",
        "semantic_score_label": "Semantic Score",
        "citation_count_label": "Citations",
        "pubmed_rank_label": "PubMed Original Rank",
        "source_label": "Source",
        "source_core": "Core Search",
        "source_related": "Related Articles",
        "quality_panel_title": "Search Quality Assessment",
        "quality_total": "Total Results",
        "quality_avg_score": "Avg Semantic Score",
        "quality_top_score": "Top Semantic Score",
        "quality_discrimination": "Discrimination",
        "quality_abstract_coverage": "Abstract Coverage",
        "quality_assessment": "Grade",
        "quality_good": "Good",
        "quality_fair": "Fair",
        "quality_poor": "Poor",
        "quality_suggestions": "Suggestions",
        "search_v3_failed": "v3.0 search failed, fell back to v1.0: {error}",
        # ---- M7 Dual Mode ----
        "mode_search": "🔍 Search Mode",
        "mode_chat": "💬 Chat Mode",
        "chat_sidebar_header": "Chat Settings",
        "chat_model_label": "Select Model",
        "chat_api_key_label": "API Key Configuration",
        "chat_provider_label": "{provider} API Key",
        "chat_provider_help": "Enter the API key for this provider (encrypted local storage)",
        "chat_save_key_btn": "Save Key",
        "chat_key_saved": "API Key saved",
        "chat_key_save_failed": "Failed to save API Key: {error}",
        "chat_title": "💬 Medical Literature Chat Assistant",
        "chat_caption": "AI-powered medical Q&A with PubMed tool calling",
        "chat_upgrade_tip": "💡 Chat mode is only available for Basic tier and above. Contact admin to upgrade.",
        "chat_discuss_btn": "Discuss these articles in Chat",
        "chat_discuss_sent": "Switched to Chat mode. Articles have been sent automatically.",
        "chat_init_failed": "Chat mode initialization failed: {error}",
        "chat_no_api_key": "No API Key available for the selected model. Please configure it in the sidebar.",
        # ---- Subscription ----
        "subscription_btn": "Upgrade Plan",
        "subscription_title": "Plans & Subscription",
        "subscription_subtitle": "Choose a plan that fits your needs and unlock more search capacity",
        "subscription_current": "Current Plan",
        "subscription_contact_btn": "Contact Admin to Upgrade",
        "subscription_contact_msg": "Please contact admin to upgrade\nEmail: admin@example.com\nWeChat: admin_wechat",
        "subscription_compare": "Feature Comparison",
        "subscription_back": "Back to Search",
        "subscription_daily_search": "Daily Searches",
        "subscription_max_articles": "Articles per Search",
        "subscription_feature_semantic": "Semantic Rerank",
        "subscription_feature_cross_encoder": "Cross-Encoder",
        "subscription_feature_citation": "Citation Boost",
        "subscription_feature_chat": "Chat Mode",
        "subscription_feature_tools": "Tool Calling",
        "subscription_feature_models": "Available Models",
    },
}


# ============================================================================
# 核心函数
# ============================================================================


def search_pubmed(keyword: str, max_results: int = 5) -> list[dict]:
    """调用 PubMed E-utilities API 检索文献，解析 XML 提取结构化数据。

    逻辑与 Coze 工作流代码节点 200001 完全一致：
    esearch 获取 PMID 列表 -> efetch 获取 XML -> ElementTree 解析 7 个字段

    Args:
        keyword: 医学研究关键词
        max_results: 检索文献数量

    Returns:
        文献字典列表，每个 dict 包含: pmid, title, abstract, authors, journal, pub_date, doi
    """
    # 1. esearch 获取 PMID 列表
    try:
        search_resp = requests.get(
            f"{EUTILS_BASE}esearch.fcgi",
            params={
                "db": "pubmed",
                "term": keyword,
                "retmax": str(max_results),
                "retmode": "json",
                "sort": "relevance",
            },
            timeout=30,
        )
        search_resp.raise_for_status()
        id_list = search_resp.json().get("esearchresult", {}).get("idlist", [])
    except Exception:
        return []

    if not id_list:
        return []

    # 2. efetch 获取文献详情（XML）
    try:
        fetch_resp = requests.get(
            f"{EUTILS_BASE}efetch.fcgi",
            params={
                "db": "pubmed",
                "id": ",".join(id_list),
                "retmode": "xml",
                "rettype": "abstract",
            },
            timeout=60,
        )
        fetch_resp.raise_for_status()
        xml_data = fetch_resp.text
    except Exception:
        return []

    # 3. 解析 XML 提取结构化数据（7 个字段）
    articles = []
    try:
        root = ET.fromstring(xml_data)
        for article_elem in root.findall(".//PubmedArticle"):
            article: dict = {}

            # PMID
            pmid_elem = article_elem.find(".//PMID")
            article["pmid"] = pmid_elem.text if pmid_elem is not None else ""

            # 标题（可能包含子标签，用 itertext 拼接）
            title_elem = article_elem.find(".//ArticleTitle")
            article["title"] = "".join(title_elem.itertext()) if title_elem is not None else ""

            # 摘要（可能多段，处理 Label 属性）
            abstract_parts = []
            for abs_elem in article_elem.findall(".//AbstractText"):
                label = abs_elem.get("Label", "")
                text = "".join(abs_elem.itertext())
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
            article["abstract"] = " ".join(abstract_parts)

            # 作者列表
            authors = []
            for author_elem in article_elem.findall(".//Author"):
                last_name = author_elem.find("LastName")
                fore_name = author_elem.find("ForeName")
                if last_name is not None and fore_name is not None:
                    authors.append(f"{fore_name.text} {last_name.text}")
                elif last_name is not None:
                    authors.append(last_name.text)
            article["authors"] = ", ".join(authors)

            # 期刊
            journal_elem = article_elem.find(".//Journal/Title")
            article["journal"] = journal_elem.text if journal_elem is not None else ""

            # 发表日期
            pub_date = article_elem.find(".//PubDate")
            if pub_date is not None:
                date_parts = []
                for tag in ("Year", "Month", "Day"):
                    el = pub_date.find(tag)
                    if el is not None:
                        date_parts.append(el.text)
                article["pub_date"] = " ".join(date_parts)
            else:
                article["pub_date"] = ""

            # DOI
            doi = ""
            for eloc in article_elem.findall(".//ELocationID"):
                if eloc.get("EIdType") == "doi":
                    doi = eloc.text or ""
                    break
            article["doi"] = doi

            articles.append(article)
    except Exception:
        pass

    return articles


def analyze_with_deepseek(
    articles: list[dict],
    keyword: str,
    analysis_type: str,
    count: int,
    api_key: str,
    lang: str = "zh",
) -> str:
    """调用 DeepSeek API 对文献进行分析，返回 Markdown 格式报告。

    三种模式（translate/summary/review）使用各自的 System Prompt，
    与 Coze 方案 docs/提示词设计/ 完全一致。

    Args:
        articles: 文献字典列表
        keyword: 检索关键词
        analysis_type: 分析模式 translate/summary/review
        count: 文献数量
        api_key: DeepSeek API Key
        lang: 界面语言 zh/en，影响 user prompt 语言

    Returns:
        Markdown 格式的分析报告
    """
    config = MODE_CONFIG.get(analysis_type, MODE_CONFIG["summary"])

    # 替换 system prompt 中的模板变量
    system_prompt = config["system_prompt"].replace("{count}", str(count))
    if analysis_type == "review":
        system_prompt = system_prompt.replace("{keyword}", keyword)

    # 构建 user prompt（语言跟随界面语言设置）
    articles_json = json.dumps(articles, ensure_ascii=False, indent=2)
    s = STRINGS.get(lang, STRINGS["zh"])
    user_prompt = s["user_prompt"].format(
        keyword=keyword,
        analysis_type=analysis_type,
        count=count,
        articles_json=articles_json,
    )

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=config["temperature"],
        max_tokens=MAX_TOKENS,
    )
    return response.choices[0].message.content


# ============================================================================
# 对话模式 - Model Adapter 包装类
# ============================================================================


class _ChatModelAdapter:
    """包装 llm.chat_completion，使其适配 chat_ui.render_chat_area 的 model_adapter 接口。

    chat_ui 要求 model_adapter 具有 chat(messages, tools?) -> LLMResponse 接口。
    本类封装模型 ID、API Key 和可选工具注册表。
    """

    def __init__(self, model_id: str, api_key: str) -> None:
        self.model_id: str = model_id
        self.api_key: str = api_key

    def chat(self, messages: list[dict], tools: Optional[list[dict]] = None) -> LLMResponse:
        """调用底层 chat_completion。

        Args:
            messages: OpenAI 格式消息列表。
            tools: 工具定义（OpenAI Function Calling 格式）。

        Returns:
            LLMResponse 对象。
        """
        return chat_completion(
            model_id=self.model_id,
            messages=messages,
            api_key=self.api_key,
            tools=tools,
        )


# ============================================================================
# 对话模式 - 侧边栏 & 主面板
# ============================================================================


def _chat_providers_for_user(user_id: int, user_tier: str) -> list[str]:
    """获取当前 tier 可用模型涉及的所有 provider 列表（去重，按使用频率排序）。

    Args:
        user_id: 用户 ID。
        user_tier: 用户 tier。

    Returns:
        provider 名称列表。
    """
    providers: list[str] = []
    seen: set[str] = set()
    models = get_available_models(user_tier)
    for mid in models:
        try:
            provider = get_provider_for_model(mid)
        except Exception:
            continue
        if provider not in seen:
            seen.add(provider)
            providers.append(provider)
    return providers


def _render_chat_sidebar(s: dict, user_id: int, user_tier: str) -> tuple[str, Optional[str]]:
    """渲染对话模式的侧边栏：会话列表 + API Key 状态。

    Args:
        s: 当前语言的 STRINGS 字典。
        user_id: 用户 ID。
        user_tier: 用户 tier。

    Returns:
        (当前模型 ID, 可用的 API Key 或 None)。
    """
    # ---- 会话列表 ----
    render_sidebar_sessions(user_id)

    # 固定使用 deepseek-chat，不提供模型选择
    model_id = "deepseek-chat"

    # 从 db 读取已保存的 API Key（统一设置页配置）
    api_key: Optional[str] = None
    try:
        api_key = resolve_api_key(model_id=model_id, user_id=user_id, env_fallback=True)
    except Exception:
        api_key = None

    # 显示 API Key 状态
    st.sidebar.divider()
    if api_key:
        st.sidebar.caption("✅ " + s.get("chat_api_key_label", "API Key 已配置"))
    else:
        st.sidebar.warning("⚠️ " + s.get("chat_no_api_key", "未配置 API Key，请在设置中配置"))

    return model_id, api_key


def _handle_search_to_chat_transition(
    s: dict,
    user_id: int,
) -> None:
    """处理从检索模式跳转到对话模式的衔接逻辑。

    当 st.session_state["search_to_chat_payload"] 存在时：
    - 创建新会话
    - 自动发送一条包含文献信息的用户消息
    - 清理 payload
    """
    payload = st.session_state.get("search_to_chat_payload")
    if not payload:
        return

    keyword: str = payload.get("keyword", "")
    articles_text: str = payload.get("articles_text", "")

    try:
        session_id = get_or_create_session(user_id=user_id)
        st.session_state["current_session_id"] = session_id
        # 清除消息缓存以确保下一轮渲染刷新
        cache_key = f"chat_messages_{session_id}"
        st.session_state.pop(cache_key, None)

        user_message = (
            f"请帮我分析以下文献：\n\n"
            f"**检索关键词：** {keyword}\n\n"
            f"**文献列表：**\n{articles_text}\n\n"
            f"请从研究目的、方法、主要发现、结论等角度进行分析。"
        )
        append_message(
            session_id=session_id,
            role="user",
            content=user_message,
        )
        st.success(s["chat_discuss_sent"])
    except Exception as e:
        st.error(s["chat_init_failed"].format(error=e))
    finally:
        st.session_state.pop("search_to_chat_payload", None)


def _render_chat_mode(
    s: dict,
    lang: str,
    user_id: int,
    user_tier: str,
) -> None:
    """渲染对话模式主界面。

    包含：标题、侧边栏（会话/模型/API Key）、主对话区。
    初始化失败时优雅降级，显示错误但不崩溃。
    """
    # free 用户不开放对话模式（理论上模式切换不会到这里，双重保险）
    if user_tier == db.TIER_FREE:
        st.info(s["chat_upgrade_tip"])
        return

    try:
        model_id, api_key = _render_chat_sidebar(s, user_id, user_tier)
    except Exception as e:
        st.error(s["chat_init_failed"].format(error=e))
        return

    # 处理检索模式跳转过来的衔接
    _handle_search_to_chat_transition(s, user_id)

    st.title(s["chat_title"])
    st.caption(s["chat_caption"])
    st.caption(s["welcome"].format(username=st.session_state["username"]))

    if not api_key:
        st.warning(s["chat_no_api_key"])
        return

    # 构建 model_adapter
    model_adapter: Optional[_ChatModelAdapter] = None
    try:
        model_adapter = _ChatModelAdapter(model_id=model_id, api_key=api_key)
    except Exception as e:
        st.error(s["chat_init_failed"].format(error=e))
        return

    # 工具注册表
    tool_registry: Optional[dict] = TOOL_REGISTRY if isinstance(TOOL_REGISTRY, dict) and TOOL_REGISTRY else None

    session_id = st.session_state.get("current_session_id", "")
    try:
        render_chat_area(
            session_id=session_id,
            user_id=user_id,
            model_adapter=model_adapter,
            tool_registry=tool_registry,
        )
    except Exception as e:
        st.error(s["chat_init_failed"].format(error=e))


def _render_mode_switch(s: dict, user_tier: str) -> str:
    """顶部模式切换（检索 / 对话）。

    使用按钮组实现，避免 radio widget 的状态同步问题。
    当前模式存 st.session_state["app_mode"]。

    Args:
        s: STRINGS 字典。
        user_tier: 用户 tier。

    Returns:
        当前模式："search" 或 "chat"。
    """
    if "app_mode" not in st.session_state:
        st.session_state["app_mode"] = "search"

    has_chat = user_tier != db.TIER_FREE
    current_mode = st.session_state["app_mode"]

    if not has_chat:
        return "search"

    # 用按钮组实现模式切换
    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            s["mode_search"],
            use_container_width=True,
            type="primary" if current_mode == "search" else "secondary",
            key="btn_mode_search",
        ):
            st.session_state["app_mode"] = "search"
            st.rerun()
    with col2:
        if st.button(
            s["mode_chat"],
            use_container_width=True,
            type="primary" if current_mode == "chat" else "secondary",
            key="btn_mode_chat",
        ):
            st.session_state["app_mode"] = "chat"
            st.rerun()

    return st.session_state["app_mode"]


# ============================================================================
# Streamlit UI
# ============================================================================


def _on_discuss_in_chat() -> None:
    """on_click 回调：点击"去对话"按钮时在脚本执行前运行。"""
    articles = st.session_state.get("_last_search_articles", [])
    keyword = st.session_state.get("_last_search_keyword", "")

    # 调试标记
    st.session_state["_debug_callback_called"] = True

    parts: list[str] = []
    for idx, art in enumerate(articles, 1):
        title = getattr(art, "title", "")
        pmid = getattr(art, "pmid", "")
        journal = getattr(art, "journal", "")
        pub_date = getattr(art, "pub_date", "")
        abstract = getattr(art, "abstract", "")
        parts.append(
            f"{idx}. **{title}** (PMID: {pmid})\n"
            f"   - 期刊: {journal}\n"
            f"   - 日期: {pub_date}\n"
            f"   - 摘要: {abstract[:500]}{'...' if abstract and len(abstract) > 500 else ''}"
        )

    st.session_state["search_to_chat_payload"] = {
        "keyword": keyword,
        "articles_text": "\n\n".join(parts),
    }
    st.session_state["app_mode"] = "chat"
    st.session_state["show_settings"] = False
    st.session_state["show_subscription"] = False


def _render_search_v3(
    s: dict,
    lang: str,
    keyword: str,
    max_results: int,
    analysis_type: str,
    api_key: str,
    use_semantic: bool = True,
    use_cross_encoder: bool = True,
    use_citations: bool = True,
    candidate_pool_size: int = 50,
) -> None:
    """v3.0 混合语义检索：调用 search.pipeline.search() 并展示结果。

    包含：进度可视化、检索策略展开、文献卡片、质量自评面板、DeepSeek 分析。
    若 v3 失败则降级到 v1.0 的 search_pubmed()。
    """
    keyword_clean = keyword.strip()
    api_key_clean = api_key.strip()

    if not keyword_clean:
        st.warning(s["warn_empty_keyword"])
        return

    if not api_key_clean:
        st.error(s["error_no_api_key"])
        return

    # ---- 进度占位 ----
    status_placeholder = st.empty()

    def _progress_cb(msg: str) -> None:
        status_placeholder.caption(f"⏳ {msg}")

    # ---- 执行 v3 检索 ----
    result: SearchResult | None = None
    try:
        if use_semantic:
            result = v3_search(
                user_input=keyword_clean,
                deepseek_api_key=api_key_clean,
                ncbi_api_key="",
                max_results=max_results,
                candidate_pool_size=candidate_pool_size,
                expand_related=False,
                use_cross_encoder=use_cross_encoder,
                use_citations=use_citations,
                progress_callback=_progress_cb,
            )
        else:
            # 用户关闭语义精排 -> 降级到 v1.0
            articles = search_pubmed(keyword_clean, max_results)
            result = SearchResult(articles=[])
            result.articles = []
            from search.models import Article as _Article
            for a in articles:
                result.articles.append(
                    _Article(
                        pmid=a.get("pmid", ""),
                        title=a.get("title", ""),
                        abstract=a.get("abstract", ""),
                        authors=a.get("authors", ""),
                        journal=a.get("journal", ""),
                        pub_date=a.get("pub_date", ""),
                        doi=a.get("doi", ""),
                        source="core",
                        pubmed_rank=0,
                        citation_count=0,
                        semantic_score=0.0,
                        rerank_score=0.0,
                        final_score=0.0,
                        has_abstract=bool(a.get("abstract")),
                    )
                )
            result.quality = {
                "total": len(result.articles),
                "avg_score": 0.0,
                "top_score": 0.0,
                "discrimination": 0.0,
                "abstract_coverage": sum(1 for a in result.articles if a.has_abstract) / len(result.articles) if result.articles else 0,
                "assessment": "fair",
                "suggestions": ["当前未启用语义精排，排序为 PubMed 原生相关度"],
            }
    except Exception as e:
        status_placeholder.empty()
        st.warning(s["search_v3_failed"].format(error=e))
        # 降级到 v1.0
        _render_search(s, lang, keyword_clean, max_results, analysis_type, api_key_clean)
        return

    status_placeholder.empty()

    if not result.articles:
        st.error(s["error_no_results"])
        st.info(s["info_keyword_tip"])
        return

    st.success(s["success_found"].format(count=len(result.articles)))

    # ---- 检索策略展开器 ----
    with st.expander(s["strategy_expander"], expanded=False):
        if result.strategy:
            st.markdown(f"**{s['strategy_boolean']}:**")
            st.code(result.strategy.boolean_query, language="text")
            st.markdown(f"**{s['strategy_semantic']}:**")
            st.code(result.strategy.semantic_query, language="text")
            if result.strategy.fallback:
                st.caption(s["strategy_fallback"])
        else:
            st.caption(s["strategy_fallback"])

    # ---- 文献卡片 ----
    st.subheader(s["article_cards_header"])
    for i, art in enumerate(result.articles, 1):
        pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{art.pmid}/"
        with st.container(border=True):
            col_title, col_score = st.columns([4, 1])
            with col_title:
                st.markdown(f"**{i}. [{art.title}]({pubmed_url})**")
            with col_score:
                if use_semantic:
                    st.progress(min(max(art.final_score, 0.0), 1.0))
                    st.caption(f"{s['semantic_score_label']}: {art.final_score:.3f}")

            meta_parts: list[str] = [f"PMID: {art.pmid}", art.journal, art.pub_date]
            if art.doi:
                meta_parts.append(f"DOI: {art.doi}")
            st.caption(" | ".join(meta_parts))

            # 评分 / 被引 / 排名 行
            info_cols = st.columns(4)
            with info_cols[0]:
                st.metric(s["semantic_score_label"], f"{art.semantic_score:.3f}")
            with info_cols[1]:
                st.metric(s["citation_count_label"], art.citation_count)
            with info_cols[2]:
                st.metric(s["pubmed_rank_label"], f"#{art.pubmed_rank + 1}")
            with info_cols[3]:
                src = s["source_core"] if art.source == "core" else s["source_related"]
                st.metric(s["source_label"], src)

            if art.abstract:
                with st.expander(s["view_abstract"], expanded=False):
                    st.text(art.abstract)

    # ---- 检索质量自评面板 ----
    if result.quality:
        with st.expander(f"📊 {s['quality_panel_title']}", expanded=False):
            q = result.quality
            col_q1, col_q2, col_q3 = st.columns(3)
            with col_q1:
                st.metric(s["quality_total"], q.get("total", 0))
                st.metric(s["quality_avg_score"], q.get("avg_score", 0))
            with col_q2:
                st.metric(s["quality_top_score"], q.get("top_score", 0))
                st.metric(s["quality_discrimination"], q.get("discrimination", 0))
            with col_q3:
                st.metric(s["quality_abstract_coverage"], f"{q.get('abstract_coverage', 0)*100:.0f}%")
                assessment = q.get("assessment", "fair")
                label = {
                    "good": s["quality_good"],
                    "fair": s["quality_fair"],
                    "poor": s["quality_poor"],
                }.get(assessment, assessment)
                st.metric(s["quality_assessment"], label)

            suggestions = q.get("suggestions", [])
            if suggestions:
                st.markdown(f"**💡 {s['quality_suggestions']}:**")
                for sug in suggestions:
                    st.markdown(f"- {sug}")

    # ---- 转换为旧格式，调用 DeepSeek 分析 ----
    articles_dicts: list[dict] = [a.to_dict() for a in result.articles]

    # 存入 session_state 供"去对话"按钮的 on_click 回调使用
    st.session_state["_last_search_articles"] = result.articles
    st.session_state["_last_search_keyword"] = keyword_clean

    with st.spinner(s["spinner_analyzing"]):
        try:
            report = analyze_with_deepseek(
                articles=articles_dicts,
                keyword=keyword_clean,
                analysis_type=analysis_type,
                count=len(articles_dicts),
                api_key=api_key_clean,
                lang=lang,
            )
        except Exception as e:
            st.error(s["error_api_failed"].format(error=e))
            st.info(s["info_retry"])
            return

    if not report or not report.strip():
        st.error(s["error_empty_response"])
        return

    # 缓存报告到 session_state（供 rerun 后重新展示）
    st.session_state["_last_search_report"] = report
    st.session_state["_last_search_analysis_type"] = analysis_type

    st.markdown("---")
    st.markdown(report)

    st.download_button(
        label=s["download_btn"],
        data=report.encode("utf-8"),
        file_name=f"pubmed_report_{keyword_clean[:20]}.md",
        mime="text/markdown",
    )

    st.divider()
    st.caption(s["footer"])

    db.log_usage(
        user_id=st.session_state["user_id"],
        keyword=keyword_clean,
        analysis_type=analysis_type,
        article_count=len(articles_dicts),
    )


def _render_search(s, lang, keyword, max_results, analysis_type, api_key):
    """执行检索并展示结果。供 main_interface 和 admin_panel 共用。"""
    # 边界场景：空关键词
    if not keyword.strip():
        st.warning(s["warn_empty_keyword"])
        return

    # 边界场景：API Key 缺失
    if not api_key.strip():
        st.error(s["error_no_api_key"])
        return

    # Step 1: PubMed 检索
    with st.spinner(s["spinner_searching"]):
        articles = search_pubmed(keyword.strip(), max_results)

    # 边界场景：检索结果为 0
    if not articles:
        st.error(s["error_no_results"])
        st.info(s["info_keyword_tip"])
        return

    st.success(s["success_found"].format(count=len(articles)))

    # 展示文献列表（含原文链接和英文摘要）
    st.subheader(s["article_list_header"])
    for i, art in enumerate(articles, 1):
        pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{art['pmid']}/"
        st.markdown(f"**{i}. [{art['title']}]({pubmed_url})**")
        meta_parts = [f"PMID: {art['pmid']}", art["journal"], art["pub_date"]]
        if art["doi"]:
            meta_parts.append(f"DOI: {art['doi']}")
        st.caption(" | ".join(meta_parts))
        if art["abstract"]:
            with st.expander(s["view_abstract"], expanded=False):
                st.text(art["abstract"])

    # Step 2: DeepSeek 分析
    with st.spinner(s["spinner_analyzing"]):
        try:
            report = analyze_with_deepseek(
                articles=articles,
                keyword=keyword.strip(),
                analysis_type=analysis_type,
                count=len(articles),
                api_key=api_key.strip(),
                lang=lang,
            )
        except Exception as e:
            st.error(s["error_api_failed"].format(error=e))
            st.info(s["info_retry"])
            return

    # 边界场景：API 返回空内容
    if not report or not report.strip():
        st.error(s["error_empty_response"])
        return

    # Step 3: 展示报告
    st.markdown("---")
    st.markdown(report)

    # 下载按钮
    st.download_button(
        label=s["download_btn"],
        data=report.encode("utf-8"),
        file_name=f"pubmed_report_{keyword.strip()[:20]}.md",
        mime="text/markdown",
    )

    # 参考链接
    st.divider()
    st.caption(s["footer"])

    # 记录用量
    db.log_usage(
        user_id=st.session_state["user_id"],
        keyword=keyword.strip(),
        analysis_type=analysis_type,
        article_count=len(articles),
    )


def _render_lang_switch(s, lang):
    """侧边栏语言切换器，返回当前语言。"""
    lang_options = {"中文": "zh", "English": "en"}
    lang_display = st.selectbox(
        s["lang_label"],
        options=list(lang_options.keys()),
        index=list(lang_options.values()).index(lang),
    )
    new_lang = lang_options[lang_display]
    if new_lang != lang:
        st.session_state["lang"] = new_lang
        st.rerun()
    return lang


def _logout():
    """清除登录状态并刷新。"""
    for key in ("logged_in", "user_id", "username", "is_admin"):
        st.session_state.pop(key, None)
    st.rerun()


def _inject_custom_css() -> None:
    """注入医疗专业蓝主题 CSS。"""
    st.markdown("""
    <style>
    /* ---- 全局字体与背景 ---- */
    .stApp {
        font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f0f4f8 0%, #e3eaf3 100%);
    }

    /* ---- 主标题区 ---- */
    .stTitle {
        color: #1a3a6c;
    }
    .stTitle h1 {
        font-weight: 700;
        border-bottom: 3px solid #2563eb;
        padding-bottom: 10px;
    }

    /* ---- 卡片容器 ---- */
    .stCard, div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stHeading"]) {
        background: white;
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border: 1px solid #e2e8f0;
    }

    /* ---- 按钮 ---- */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        border: none;
        transition: all 0.2s;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2563eb 0%, #1a3a6c 100%);
        color: white;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #1e40af 0%, #162d50 100%);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }

    /* ---- 侧边栏 ---- */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #1a3a6c;
    }

    /* ---- Tab 样式 ---- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
        background-color: #f1f5f9;
        border: 1px solid #e2e8f0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563eb !important;
        color: white !important;
    }

    /* ---- 进度条 ---- */
    .stProgress > div > div {
        background: linear-gradient(90deg, #2563eb 0%, #3b82f6 100%);
    }

    /* ---- Metric 卡片 ---- */
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    /* ---- 登录页 ---- */
    .stApp [data-testid="stVerticalBlock"] {
        gap: 12px;
    }

    /* ---- 展开器 ---- */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #1a3a6c;
        background-color: #f0f4f8;
        border-radius: 8px;
    }

    /* ---- chat 消息 ---- */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)


def subscription_page(s: dict) -> None:
    """套餐订阅页面：展示 5 档套餐卡片、功能对比表、升级引导。"""
    user_tier = db.get_user_tier(st.session_state["user_id"])
    current_name = tier_gating.get_tier_display_name(user_tier)

    # ---- 自定义标题栏 ----
    st.markdown(
        '<div style="background: linear-gradient(135deg, #1a3a6c 0%, #2563eb 100%); '
        'border-radius: 12px; padding: 24px 32px; margin-bottom: 20px;">'
        f'<h1 style="color: white; margin: 0; font-weight: 700;">{s["subscription_title"]}</h1>'
        f'<p style="color: #c7d6f0; margin: 8px 0 0 0;">{s["subscription_subtitle"]}</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.caption(f'{s["subscription_current"]}: **{current_name}**')

    tier_order = [
        db.TIER_FREE, db.TIER_BASIC, db.TIER_PRO,
        db.TIER_FLAGSHIP, db.TIER_INSTITUTIONAL,
    ]

    # ---- 套餐卡片 ----
    cols = st.columns(len(tier_order))
    for col, tier_key in zip(cols, tier_order):
        feat = tier_gating.TIER_FEATURES[tier_key]
        is_current = tier_key == user_tier
        border_color = "#2563eb" if is_current else "#e2e8f0"
        bg_color = "#eff6ff" if is_current else "white"
        name = feat["name"]
        daily = feat["daily_search_limit"]
        articles = feat["max_articles_per_search"]
        with col:
            st.markdown(
                f'<div style="border: 2px solid {border_color}; background: {bg_color}; '
                f'border-radius: 12px; padding: 16px; margin-bottom: 8px;">'
                f'<h3 style="color: #1a3a6c; margin: 0;">{name}</h3>'
                '</div>',
                unsafe_allow_html=True,
            )
            st.markdown(f'- {s["subscription_daily_search"]}: **{daily}**')
            st.markdown(f'- {s["subscription_max_articles"]}: **{articles}**')
            st.markdown(f'- {s["subscription_feature_semantic"]}: {"✅" if feat["semantic_rerank"] else "❌"}')
            st.markdown(f'- {s["subscription_feature_cross_encoder"]}: {"✅" if feat["cross_encoder"] else "❌"}')
            st.markdown(f'- {s["subscription_feature_citation"]}: {"✅" if feat["citation_boost"] else "❌"}')
            st.markdown(f'- {s["subscription_feature_chat"]}: {"✅" if feat["chat_mode"] else "❌"}')
            if is_current:
                st.success(s["subscription_current"])
            else:
                if st.button(s["subscription_contact_btn"], key=f"sub_{tier_key}", use_container_width=True):
                    st.info(s["subscription_contact_msg"])

    # ---- 功能对比表 ----
    st.divider()
    st.subheader(s["subscription_compare"])
    headers = [
        s["subscription_daily_search"], s["subscription_max_articles"],
        s["subscription_feature_semantic"], s["subscription_feature_cross_encoder"],
        s["subscription_feature_citation"], s["subscription_feature_chat"],
    ]
    lines = ["| 套餐 | " + " | ".join(headers) + " |",
             "|" + "---|" * (len(headers) + 1)]
    for tier_key in tier_order:
        feat = tier_gating.TIER_FEATURES[tier_key]
        label = feat["name"]
        if tier_key == user_tier:
            label += f" ({s['subscription_current']})"
        row = [
            label,
            str(feat["daily_search_limit"]),
            str(feat["max_articles_per_search"]),
            "✅" if feat["semantic_rerank"] else "❌",
            "✅" if feat["cross_encoder"] else "❌",
            "✅" if feat["citation_boost"] else "❌",
            "✅" if feat["chat_mode"] else "❌",
        ]
        lines.append("| " + " | ".join(row) + " |")
    st.markdown("\n".join(lines))

    # ---- 返回按钮 ----
    if st.button(s["subscription_back"], type="primary"):
        st.session_state["show_subscription"] = False
        st.rerun()


def login_page(s):
    """登录页面。"""
    # 顶部 logo / 标题区（医疗专业蓝风格）
    st.markdown(
        '<div style="background: linear-gradient(135deg, #1a3a6c 0%, #2563eb 100%); '
        'border-radius: 12px; padding: 30px 36px; margin-bottom: 24px; text-align: center;">'
        '<h1 style="color: white; margin: 0; font-weight: 700; font-size: 28px;">📚 '
        f'{s["login_title"]}</h1>'
        f'<p style="color: #c7d6f0; margin: 10px 0 0 0;">{s["caption"]}</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        _render_lang_switch(s, st.session_state["lang"])

    username = st.text_input(s["login_username"])
    password = st.text_input(s["login_password"], type="password")

    if st.button(s["login_btn"], type="primary"):
        user = db.verify_user(username, password)
        if user:
            st.session_state["logged_in"] = True
            st.session_state["user_id"] = user["id"]
            st.session_state["username"] = user["username"]
            st.session_state["is_admin"] = user["is_admin"]
            st.rerun()
        else:
            st.error(s["login_failed"])

    st.info(s["login_hint"])


def admin_panel(s):
    """管理后台：客户管理、用量统计、最近检索、检索测试。"""
    lang = st.session_state["lang"]

    st.title(s["admin_panel_title"])
    st.caption(s["welcome"].format(username=st.session_state["username"]))

    # 返回按钮
    if st.button("← 返回主界面"):
        st.session_state["show_admin"] = False
        st.rerun()

    tab_users, tab_stats, tab_recent, tab_tier, tab_search = st.tabs([
        s["admin_user_list"],
        s["admin_usage_stats"],
        s["admin_recent_usage"],
        "Tier 管理",
        s["admin_search_test"],
    ])

    # ---- 客户管理 ----
    with tab_users:
        with st.form("add_user_form"):
            st.subheader(s["admin_add_user"])
            new_username = st.text_input(s["admin_new_username"])
            new_password = st.text_input(s["admin_new_password"], type="password")
            if st.form_submit_button(s["admin_create_btn"]):
                ok, msg = db.create_user(new_username, new_password)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

        st.divider()

        st.subheader(s["admin_user_list"])
        users = db.get_all_users()
        for u in users:
            role = s["admin_admin_role"] if u["is_admin"] else s["admin_user_role"]
            status = s["admin_active"] if u["is_active"] else s["admin_disabled"]
            col1, col2, col3 = st.columns([4, 1, 1])
            with col1:
                st.markdown(f"**{u['username']}** | {role} | {status}")
                st.caption(f"ID: {u['id']} | {u['created_at']}")
            with col2:
                if not u["is_admin"]:
                    toggle_label = s["admin_disabled"] if u["is_active"] else s["admin_active"]
                    if st.button(toggle_label, key=f"toggle_{u['id']}"):
                        db.toggle_user_active(u["id"], not u["is_active"])
                        st.rerun()
            with col3:
                if st.button(s["admin_delete"], key=f"del_{u['id']}"):
                    if u["id"] == st.session_state["user_id"]:
                        st.error(s["admin_no_delete_self"])
                    elif u["is_admin"]:
                        st.error(s["admin_no_delete_admin"])
                    else:
                        db.delete_user(u["id"])
                        st.rerun()

        st.divider()

        st.subheader(s["admin_reset_pwd"])
        user_options = {f"{u['username']} (ID:{u['id']})": u["id"] for u in users}
        selected_label = st.selectbox(
            s["admin_user_list"], options=list(user_options.keys())
        )
        selected_id = user_options[selected_label]
        new_pwd = st.text_input(s["admin_new_password_prompt"], type="password")
        if st.button(s["admin_reset_pwd"]):
            ok, msg = db.reset_password(selected_id, new_pwd)
            if ok:
                st.success(msg)
            else:
                st.error(msg)

    # ---- 用量统计 ----
    with tab_stats:
        st.subheader(s["admin_usage_stats"])
        stats = db.get_all_usage_stats()
        if not stats:
            st.info(s["no_history"])
        else:
            headers = [
                s["login_username"], s["admin_role"], s["admin_status"],
                s["admin_total_searches"], s["admin_total_articles"], s["admin_last_search"],
            ]
            lines = ["| " + " | ".join(headers) + " |",
                     "|" + "|".join(["---"] * len(headers)) + "|"]
            for stat in stats:
                role = s["admin_admin_role"] if stat["is_admin"] else s["admin_user_role"]
                status = s["admin_active"] if stat["is_active"] else s["admin_disabled"]
                lines.append(
                    f"| {stat['username']} | {role} | {status} "
                    f"| {stat['total_searches']} | {stat['total_articles']} "
                    f"| {stat['last_search'] or '-'} |"
                )
            st.markdown("\n".join(lines))

    # ---- 最近检索 ----
    with tab_recent:
        st.subheader(s["admin_recent_usage"])
        recent = db.get_recent_usage()
        if not recent:
            st.info(s["no_history"])
        else:
            headers = [
                s["login_username"], s["keyword_label"], s["analysis_label"],
                s["max_results_label"], s["history_time"],
            ]
            lines = ["| " + " | ".join(headers) + " |",
                     "|" + "|".join(["---"] * len(headers)) + "|"]
            for r in recent:
                lines.append(
                    f"| {r['username']} | {r['keyword']} | {r['analysis_type']} "
                    f"| {r['article_count']} | {r['created_at']} |"
                )
            st.markdown("\n".join(lines))

    # ---- Tier 管理 ----
    with tab_tier:
        st.subheader("用户 Tier 管理")
        st.caption("控制用户的功能权限等级（Cross-Encoder 默认值等）")
        all_users = db.get_all_users()
        tier_options = [
            db.TIER_FREE,
            db.TIER_BASIC,
            db.TIER_PRO,
            db.TIER_FLAGSHIP,
            db.TIER_INSTITUTIONAL,
        ]
        for u in all_users:
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                role = s["admin_admin_role"] if u["is_admin"] else s["admin_user_role"]
                st.markdown(f"**{u['username']}** | ID:{u['id']} | {role}")
            with col2:
                current_tier = db.get_user_tier(u["id"])
                new_tier = st.selectbox(
                    "Tier",
                    options=tier_options,
                    index=tier_options.index(current_tier) if current_tier in tier_options else 0,
                    key=f"tier_select_{u['id']}",
                    label_visibility="collapsed",
                )
            with col3:
                if st.button("保存", key=f"tier_save_{u['id']}"):
                    ok, msg = db.set_user_tier(u["id"], new_tier)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)

    # ---- 检索测试 ----
    with tab_search:
        st.subheader(s["admin_search_test"])
        keyword = st.text_input(
            s["keyword_label"],
            placeholder="myocardial infarction treatment",
            help=s["keyword_help"],
        )
        max_results = st.slider(s["max_results_label"], min_value=3, max_value=200, value=10)
        analysis_type = st.selectbox(
            s["analysis_label"],
            options=["translate", "summary", "review"],
            format_func=lambda x: {
                "translate": s["analysis_translate"],
                "summary": s["analysis_summary"],
                "review": s["analysis_review"],
            }[x],
        )
        # API Key 输入（默认为空，客户自行输入）
        api_key = st.text_input(
            s["api_key_label"],
            value="",
            type="password",
            help=s["api_key_input_help"],
        )
        if st.button(s["run_btn"], type="primary"):
            _render_search(s, lang, keyword, max_results, analysis_type, api_key)


def _render_settings_page(s: dict, user_id: int) -> None:
    """统一设置页：API Key 配置。

    用户在此配置一次 DeepSeek API Key，检索模式和对话模式共用。
    """
    st.markdown(
        '<div style="background: linear-gradient(135deg, #1a3a6c 0%, #2563eb 100%); '
        'border-radius: 12px; padding: 24px 32px; margin-bottom: 20px;">'
        '<h2 style="color: white; margin: 0; font-weight: 700;">⚙️ 设置</h2>'
        '<p style="color: #c7d6f0; margin: 8px 0 0 0;">配置 API Key，检索模式和对话模式共用</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ---- DeepSeek API Key ----
    st.subheader("DeepSeek API Key")

    # 检查当前是否已配置
    saved_key = db.get_user_api_key(user_id, "deepseek")
    env_key = os.environ.get("DEEPSEEK_API_KEY", "")

    if saved_key:
        st.success("✅ 已配置 DeepSeek API Key（已加密保存）")
        # 显示 key 的前几位和后几位
        masked = saved_key[:6] + "*" * 20 + saved_key[-4:] if len(saved_key) > 10 else "****"
        st.caption(f"当前 Key: {masked}")
    elif env_key:
        st.info("📌 当前使用环境变量中的 API Key，如需更换可在此输入")
    else:
        st.warning("⚠️ 尚未配置 DeepSeek API Key，请在此输入")

    # 输入新 Key
    new_key = st.text_input(
        "输入 DeepSeek API Key",
        value="",
        type="password",
        placeholder="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        help="API Key 将加密保存到本地数据库，刷新页面不会丢失。获取地址：https://platform.deepseek.com/",
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("💾 保存", type="primary", use_container_width=True):
            if new_key.strip():
                try:
                    db.set_user_api_key(user_id, "deepseek", new_key.strip())
                    st.success("✅ API Key 已保存！检索和对话模式均可使用。")
                    st.rerun()
                except Exception as e:
                    st.error(f"保存失败: {e}")
            else:
                st.error("请输入 API Key")
    with col2:
        if saved_key and st.button("🗑️ 清除已保存的 Key"):
            try:
                db.delete_user_api_key(user_id, "deepseek")
                st.success("已清除保存的 API Key")
                st.rerun()
            except Exception as e:
                st.error(f"清除失败: {e}")

    st.divider()
    st.markdown("### 获取 API Key")
    st.markdown(
        "- **DeepSeek**: 访问 [https://platform.deepseek.com/](https://platform.deepseek.com/) 注册并创建 API Key\n"
        "- 将 Key 粘贴到上方输入框，点击保存\n"
        "- 保存后刷新页面无需重新输入"
    )

    st.divider()
    if st.button("← 返回", use_container_width=True):
        st.session_state["show_settings"] = False
        st.rerun()


def main_interface(s):
    """主界面：双模式（检索 / 对话）+ 个人历史（普通客户）。"""
    lang = st.session_state["lang"]
    user_tier = db.get_user_tier(st.session_state["user_id"])

    # ---- 模式控制：读 session_state ----
    if "app_mode" not in st.session_state:
        st.session_state["app_mode"] = "search"

    # ---- 顶部模式切换 ----
    app_mode = _render_mode_switch(s, user_tier)

    # ---- 对话模式 ----
    if app_mode == "chat":
        _render_chat_mode(s, lang, st.session_state["user_id"], user_tier)
        return

    # ---- 检索模式 ----
    # 自定义 HTML 标题栏（医疗专业蓝渐变背景）
    st.markdown(
        '<div style="background: linear-gradient(135deg, #1a3a6c 0%, #2563eb 100%); '
        'border-radius: 12px; padding: 24px 32px; margin-bottom: 16px;">'
        f'<h1 style="color: white; margin: 0; font-weight: 700;">{s["title"]}</h1>'
        f'<p style="color: #c7d6f0; margin: 8px 0 0 0;">{s["caption"]}</p>'
        f'<p style="color: #c7d6f0; margin: 4px 0 0 0;">{s["welcome"].format(username=st.session_state["username"])}</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Cross-Encoder 默认值：pro 及以上默认开启，以下默认关闭
    _pro_plus = user_tier in (db.TIER_PRO, db.TIER_FLAGSHIP, db.TIER_INSTITUTIONAL)

    with st.sidebar:
        _render_lang_switch(s, lang)

        st.header(s["sidebar_header"])

        keyword = st.text_input(
            s["keyword_label"],
            placeholder="myocardial infarction treatment",
            help=s["keyword_help"],
        )

        max_articles = tier_gating.get_max_articles(user_tier)
        max_results = st.slider(
            s["max_results_label"],
            min_value=3,
            max_value=max_articles,
            value=min(10, max_articles),
        )

        analysis_type = st.selectbox(
            s["analysis_label"],
            options=["translate", "summary", "review"],
            format_func=lambda x: {
                "translate": s["analysis_translate"],
                "summary": s["analysis_summary"],
                "review": s["analysis_review"],
            }[x],
        )

        # ---- v3.0 语义检索设置 ----
        st.divider()
        st.subheader(s["semantic_header"])

        use_semantic = st.toggle(
            s["semantic_rerank_label"],
            value=True,
            help=s["semantic_rerank_help"],
        )

        use_cross_encoder = st.toggle(
            s["cross_encoder_label"],
            value=_pro_plus,
            help=s["cross_encoder_help"],
        )

        use_citations = st.toggle(
            s["citation_label"],
            value=True,
            help=s["citation_help"],
        )

        candidate_pool_size = st.slider(
            s["candidate_pool_label"],
            min_value=20,
            max_value=max(100, max_articles * 3),
            value=min(max(100, max_articles * 3), max(50, max_articles * 2)),
            step=10,
            help=s["candidate_pool_help"],
        )

        st.divider()

        # API Key 从 db 读取（统一设置页配置）
        user_id = st.session_state["user_id"]
        api_key = db.get_user_api_key(user_id, "deepseek") or os.environ.get("DEEPSEEK_API_KEY", "")
        if api_key:
            st.caption("✅ API Key 已配置")
        else:
            st.warning("⚠️ 未配置 API Key，请点击下方「设置」")

        run_btn = st.button(s["run_btn"], type="primary", use_container_width=True)

        st.divider()
        st.caption(s["cost_note"])

        # 个人检索历史
        with st.expander(s["my_history"]):
            history = db.get_usage_by_user(st.session_state["user_id"])
            if not history:
                st.info(s["no_history"])
            else:
                for h in history:
                    st.text(
                        f"{h['created_at']} | {h['keyword']} "
                        f"| {h['analysis_type']} | {h['article_count']}"
                    )

        st.divider()
        if st.button("⚙️ 设置", use_container_width=True):
            st.session_state["show_settings"] = True
            st.session_state["show_subscription"] = False
            st.rerun()

        if st.button(s["subscription_btn"], use_container_width=True):
            st.session_state["show_subscription"] = True
            st.session_state["show_settings"] = False
            st.rerun()

        # admin 用户显示管理入口
        if st.session_state.get("is_admin"):
            st.divider()
            if st.button("🛠️ 管理后台", use_container_width=True):
                st.session_state["show_admin"] = True
                st.session_state["show_settings"] = False
                st.session_state["show_subscription"] = False
                st.rerun()

        st.divider()
        if st.button(s["logout_btn"], use_container_width=True):
            _logout()

    # 主面板：管理后台 / 设置页 / 订阅页 / 检索结果
    if st.session_state.get("show_admin") and st.session_state.get("is_admin"):
        admin_panel(s)
    elif st.session_state.get("show_settings"):
        _render_settings_page(s, st.session_state["user_id"])
    elif st.session_state.get("show_subscription"):
        subscription_page(s)
    elif run_btn:
        _render_search_v3(
            s, lang, keyword, max_results, analysis_type, api_key,
            use_semantic=use_semantic,
            use_cross_encoder=use_cross_encoder,
            use_citations=use_citations,
            candidate_pool_size=candidate_pool_size,
        )

    # ---- "去对话"按钮（放在 run_btn 块之外，每次 rerun 都渲染）----
    # 只要上次检索结果存在且用户有对话权限，就显示按钮
    _cached_articles = st.session_state.get("_last_search_articles")
    _cached_report = st.session_state.get("_last_search_report")
    if _cached_articles and user_tier != db.TIER_FREE:
        # 非 run_btn 时重新展示报告（run_btn 时 _render_search_v3 已展示）
        if not run_btn and _cached_report:
            st.markdown("---")
            st.markdown(_cached_report)
            st.download_button(
                label=s["download_btn"],
                data=_cached_report.encode("utf-8"),
                file_name="pubmed_report.md",
                mime="text/markdown",
            )
        # 始终显示"去对话"按钮
        if st.button(s["chat_discuss_btn"], type="secondary", key="btn_discuss_in_chat_main"):
            # 构造 payload
            parts: list[str] = []
            for idx, art in enumerate(_cached_articles, 1):
                title = getattr(art, "title", "")
                pmid = getattr(art, "pmid", "")
                journal = getattr(art, "journal", "")
                pub_date = getattr(art, "pub_date", "")
                abstract = getattr(art, "abstract", "")
                parts.append(
                    f"{idx}. **{title}** (PMID: {pmid})\n"
                    f"   - 期刊: {journal}\n"
                    f"   - 日期: {pub_date}\n"
                    f"   - 摘要: {abstract[:500]}{'...' if abstract and len(abstract) > 500 else ''}"
                )
            st.session_state["search_to_chat_payload"] = {
                "keyword": st.session_state.get("_last_search_keyword", ""),
                "articles_text": "\n\n".join(parts),
            }
            st.session_state["app_mode"] = "chat"
            st.rerun()


def main() -> None:
    """路由函数：初始化数据库，根据登录状态分流。"""
    db.init_db()

    # 语言选择（优先用 session_state 持久化）
    if "lang" not in st.session_state:
        st.session_state["lang"] = "zh"
    lang = st.session_state["lang"]
    s = STRINGS[lang]

    st.set_page_config(
        page_title=s["page_title"],
        page_icon="📚",
        layout="wide",
    )

    # 注入医疗专业蓝主题 CSS
    _inject_custom_css()

    # 初始化登录状态
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    # 路由：admin 和普通用户都进 main_interface，admin 额外有管理入口
    if not st.session_state["logged_in"]:
        login_page(s)
    else:
        main_interface(s)


if __name__ == "__main__":
    main()
