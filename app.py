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

import requests
import streamlit as st
from openai import OpenAI

import db

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
5. 如果某篇文献无摘要，标注"该文献暂无摘要\""""

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
# Streamlit UI
# ============================================================================


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


def login_page(s):
    """登录页面。"""
    st.title(s["login_title"])

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

    with st.sidebar:
        _render_lang_switch(s, lang)
        st.divider()
        if st.button(s["logout_btn"], use_container_width=True):
            _logout()

    tab_users, tab_stats, tab_recent, tab_search = st.tabs([
        s["admin_user_list"],
        s["admin_usage_stats"],
        s["admin_recent_usage"],
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

    # ---- 检索测试 ----
    with tab_search:
        st.subheader(s["admin_search_test"])
        keyword = st.text_input(
            s["keyword_label"],
            placeholder="myocardial infarction treatment",
            help=s["keyword_help"],
        )
        max_results = st.slider(s["max_results_label"], min_value=3, max_value=10, value=5)
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


def main_interface(s):
    """主界面：检索 + 个人历史（普通客户）。"""
    lang = st.session_state["lang"]

    st.title(s["title"])
    st.caption(s["caption"])
    st.caption(s["welcome"].format(username=st.session_state["username"]))

    with st.sidebar:
        _render_lang_switch(s, lang)

        st.header(s["sidebar_header"])

        keyword = st.text_input(
            s["keyword_label"],
            placeholder="myocardial infarction treatment",
            help=s["keyword_help"],
        )

        max_results = st.slider(s["max_results_label"], min_value=3, max_value=10, value=5)

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
        if st.button(s["logout_btn"], use_container_width=True):
            _logout()

    # 主面板：检索结果
    if run_btn:
        _render_search(s, lang, keyword, max_results, analysis_type, api_key)


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

    # 初始化登录状态
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    # 路由
    if not st.session_state["logged_in"]:
        login_page(s)
    elif st.session_state.get("is_admin"):
        admin_panel(s)
    else:
        main_interface(s)


if __name__ == "__main__":
    main()
