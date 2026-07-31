"""消息渲染器 - 使用 Streamlit 原生组件渲染对话消息。

提供单条消息、文献引用卡片、工具调用状态的渲染函数。
设计风格与 search/ 模块的展示方式保持一致。
"""

from typing import Optional


def render_message(role: str, content: str, message_id: int = 0) -> None:
    """渲染单条对话消息。

    使用 st.chat_message 作为容器，内容以 Markdown 形式展示。
    tool 角色的消息会折叠显示在 expander 中。

    Args:
        role: 消息角色（user / assistant / tool / system）。
        content: 消息内容文本。
        message_id: 消息 ID，用于生成稳定的 key（可选）。
    """
    import streamlit as st  # 延迟导入，避免非 Streamlit 环境报错

    if role == "system":
        # 系统提示词不直接渲染给用户
        return

    if role == "tool":
        # 工具结果折叠显示
        with st.expander(f"🔧 工具结果 #{message_id}", expanded=False):
            st.markdown(content or "(无内容)")
        return

    avatar = "🧑‍⚕️" if role == "user" else "🤖"
    with st.chat_message(role, avatar=avatar):
        try:
            st.markdown(content or "(无内容)")
        except Exception:
            st.text(content or "(无内容)")


def render_article_citations(articles: list[dict]) -> None:
    """渲染文献引用卡片列表。

    每张卡片展示标题、期刊、PMID，点击可展开查看摘要。
    风格保持与 app.py 中文献展示一致。

    Args:
        articles: 文献字典列表，每条应包含 pmid/title/journal/abstract/authors/pub_date/doi 等字段。
    """
    import streamlit as st  # 延迟导入

    if not articles:
        return

    st.markdown(f"**📚 相关文献（共 {len(articles)} 篇）**")

    for idx, art in enumerate(articles, start=1):
        pmid = str(art.get("pmid", "N/A"))
        title = art.get("title", "无标题")
        journal = art.get("journal", "")
        pub_date = art.get("pub_date", "")
        authors = art.get("authors", "")
        abstract = art.get("abstract", "")
        doi = art.get("doi", "")

        header_parts = [f"**[{idx}] {title}**"]
        meta_parts = []
        if journal:
            meta_parts.append(journal)
        if pub_date:
            meta_parts.append(pub_date)
        if meta_parts:
            header_parts.append(f"  \n<small>{' · '.join(meta_parts)}</small>")

        header_text = "".join(header_parts)

        with st.expander(header_text, expanded=False):
            if authors:
                st.markdown(f"**作者：** {authors}")
            if pmid and pmid != "N/A":
                st.markdown(
                    f"**PMID：** [{pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)"
                )
            if doi:
                st.markdown(f"**DOI：** {doi}")
            if abstract:
                st.markdown("**摘要：**")
                st.markdown(abstract)


def render_tool_call(
    tool_name: str,
    args: dict,
    status: str = "done",
) -> None:
    """渲染工具调用状态。

    在 assistant 消息的上下文中，展示一次工具调用的名称、参数和状态。

    Args:
        tool_name: 工具名称（如 pubmed_search）。
        args: 工具参数字典。
        status: 状态，可选 pending / running / done / error。
    """
    import streamlit as st  # 延迟导入

    status_icon = {
        "pending": "⏳",
        "running": "🔄",
        "done": "✅",
        "error": "❌",
    }.get(status, "🔧")

    status_text = {
        "pending": "等待中",
        "running": "执行中",
        "done": "已完成",
        "error": "失败",
    }.get(status, status)

    label = f"{status_icon} 调用工具 `{tool_name}` - {status_text}"

    if args:
        with st.expander(label, expanded=(status == "error")):
            try:
                st.json(args)
            except Exception:
                st.text(str(args))
    else:
        st.markdown(label)
