"""chat 包 - 对话模块。

提供会话管理、上下文构建、消息渲染和 Streamlit 对话 UI 组件，
作为 Streamlit 对话界面的底层支撑。

模块结构：
- chat_manager: 会话与消息的持久化管理（底层调用 db.py）
- context_manager: 上下文构建与截断，转换为 LLM 可用的 messages 格式
- message_renderer: 消息渲染（Markdown、文献卡片、工具调用展示）
- chat_ui: Streamlit 对话 UI 组件（侧边栏、主对话区）
"""

from chat.chat_manager import (
    get_or_create_session,
    list_sessions,
    delete_session,
    rename_session,
    append_message,
    get_messages,
)
from chat.context_manager import build_context, estimate_tokens
from chat.message_renderer import (
    render_message,
    render_article_citations,
    render_tool_call,
)
from chat.chat_ui import render_sidebar_sessions, render_chat_area

__all__ = [
    # chat_manager
    "get_or_create_session",
    "list_sessions",
    "delete_session",
    "rename_session",
    "append_message",
    "get_messages",
    # context_manager
    "build_context",
    "estimate_tokens",
    # message_renderer
    "render_message",
    "render_article_citations",
    "render_tool_call",
    # chat_ui
    "render_sidebar_sessions",
    "render_chat_area",
]
