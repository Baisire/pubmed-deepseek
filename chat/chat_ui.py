"""Streamlit 对话 UI 组件。

提供侧边栏会话列表和主对话区的渲染函数，
使用 session_state 管理当前会话 ID 和消息缓存。

设计要点：
- 侧边栏：会话列表 + 新建会话按钮 + 删除/重命名入口
- 主对话区：历史消息渲染 + 输入框 + LLM 回复流式占位
- model_adapter 与 tool_registry 以 Optional 形式接入，缺失时降级为纯文本回复
"""

import json
from typing import Any, Callable, Optional

import streamlit as st

from chat.chat_manager import (
    append_message,
    delete_session,
    get_or_create_session,
    get_messages,
    list_sessions,
    rename_session,
)
from chat.context_manager import build_context
from chat.message_renderer import (
    render_article_citations,
    render_message,
    render_tool_call,
)


# ---------------------------------------------------------------------------
# 侧边栏会话管理
# ---------------------------------------------------------------------------

def render_sidebar_sessions(user_id: int) -> Optional[str]:
    """渲染侧边栏的会话列表 + 新建按钮，返回当前选中的会话 ID。

    使用 st.session_state 管理当前会话：
    - st.session_state["current_session_id"]: 当前选中的会话 ID

    Args:
        user_id: 用户 ID。

    Returns:
        当前选中的会话 ID；若没有任何会话则返回 None。
    """
    with st.sidebar:
        st.markdown("## 💬 对话会话")

        # 新建会话按钮
        if st.button("➕ 新建对话", use_container_width=True, key="new_chat_btn"):
            try:
                new_id = get_or_create_session(user_id=user_id)
                st.session_state["current_session_id"] = new_id
                st.session_state.pop("chat_messages_cache", None)
            except Exception as e:
                st.error(f"创建会话失败: {e}")

        st.divider()

        # 会话列表
        sessions = list_sessions(user_id)

        if not sessions:
            st.info("暂无对话，点击上方按钮创建新对话。")
            # 自动创建一个默认会话
            if "current_session_id" not in st.session_state:
                try:
                    new_id = get_or_create_session(user_id=user_id)
                    st.session_state["current_session_id"] = new_id
                except Exception:
                    pass
            return st.session_state.get("current_session_id")

        current_id: Optional[str] = st.session_state.get("current_session_id")

        # 若当前无选中，使用最新的一个
        if not current_id and sessions:
            current_id = sessions[0]["id"]
            st.session_state["current_session_id"] = current_id

        for sess in sessions:
            sess_id = sess["id"]
            title = sess.get("title", "新对话")
            is_active = (sess_id == current_id)

            col1, col2 = st.columns([5, 1])
            with col1:
                btn_label = f"{'▶ ' if is_active else ''}{title[:20]}{'…' if len(title) > 20 else ''}"
                if st.button(
                    btn_label,
                    key=f"session_{sess_id}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    st.session_state["current_session_id"] = sess_id
                    st.session_state.pop("chat_messages_cache", None)
                    st.rerun()

            with col2:
                if st.button("🗑️", key=f"del_{sess_id}", help="删除会话"):
                    try:
                        delete_session(sess_id)
                        if st.session_state.get("current_session_id") == sess_id:
                            st.session_state.pop("current_session_id", None)
                            st.session_state.pop("chat_messages_cache", None)
                        st.rerun()
                    except Exception as e:
                        st.error(f"删除失败: {e}")

        return st.session_state.get("current_session_id")


# ---------------------------------------------------------------------------
# 主对话区
# ---------------------------------------------------------------------------

def _execute_tool_calls(
    tool_calls: list[dict],
    tool_registry: Optional[dict],
) -> list[dict]:
    """执行工具调用并返回结果消息列表。

    Args:
        tool_calls: OpenAI 格式的工具调用列表。
        tool_registry: 工具注册表，形如 {tool_name: callable}。

    Returns:
        tool 角色的消息列表（OpenAI 格式）。
    """
    results: list[dict] = []
    if not tool_calls:
        return results

    for tc in tool_calls:
        call_id = tc.get("id", "")
        func = tc.get("function", {})
        name = func.get("name", "")
        args_str = func.get("arguments", "{}")

        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
        except (json.JSONDecodeError, TypeError):
            args = {}

        # 渲染工具调用状态（运行中）
        render_tool_call(tool_name=name, args=args, status="running")

        content = ""
        status = "done"

        if tool_registry and name in tool_registry:
            try:
                tool_fn = tool_registry[name]
                raw_result = tool_fn(**args) if isinstance(args, dict) else tool_fn(args)
                if isinstance(raw_result, str):
                    content = raw_result
                else:
                    content = json.dumps(raw_result, ensure_ascii=False, indent=2)
            except Exception as e:
                content = f"工具执行失败: {e}"
                status = "error"
        else:
            content = f"未注册的工具: {name}"
            status = "error"

        # 更新工具调用状态（完成/失败）
        render_tool_call(tool_name=name, args=args, status=status)

        results.append({
            "role": "tool",
            "content": content,
            "tool_call_id": call_id,
        })

    return results


def _generate_reply(
    messages: list[dict],
    model_adapter: Optional[Any],
    tool_registry: Optional[dict],
) -> dict:
    """调用 LLM 生成回复，支持多轮工具调用。

    若 model_adapter 缺失，则返回一条降级提示文本。

    Args:
        messages: OpenAI 格式的消息列表。
        model_adapter: 模型适配器对象，需提供 chat(messages, tools?) -> LLMResponse 接口。
        tool_registry: 工具注册表。

    Returns:
        最终 assistant 消息字典（role/content/tool_calls）。
    """
    if model_adapter is None:
        return {
            "role": "assistant",
            "content": "⚠️ 模型适配器未配置，当前无法生成回复。请检查 LLM 模块是否正确加载。",
            "tool_calls": [],
        }

    tools_schema = None
    if tool_registry:
        # 预留：从注册表提取工具 schema（格式由注册方决定，此处仅透传）
        tools_schema = getattr(tool_registry, "tools_schema", None)

    max_iterations = 5
    working_messages = list(messages)

    for _ in range(max_iterations):
        try:
            if hasattr(model_adapter, "chat"):
                response = model_adapter.chat(working_messages, tools=tools_schema)
            elif callable(model_adapter):
                response = model_adapter(working_messages, tools=tools_schema)
            else:
                return {
                    "role": "assistant",
                    "content": "⚠️ 模型适配器接口不兼容。",
                    "tool_calls": [],
                }
        except Exception as e:
            return {
                "role": "assistant",
                "content": f"⚠️ 模型调用失败: {e}",
                "tool_calls": [],
            }

        content = getattr(response, "content", "") if not isinstance(response, dict) else response.get("content", "")
        tool_calls = getattr(response, "tool_calls", []) if not isinstance(response, dict) else response.get("tool_calls", [])

        assistant_msg = {
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls or [],
        }

        if not tool_calls:
            return assistant_msg

        working_messages.append(assistant_msg)

        # 执行工具调用
        tool_results = _execute_tool_calls(tool_calls, tool_registry)
        working_messages.extend(tool_results)

    # 达到最大迭代，返回最后一条 assistant 消息
    return {
        "role": "assistant",
        "content": "⚠️ 达到最大工具调用迭代次数，停止继续调用。",
        "tool_calls": [],
    }


def _extract_articles_from_reply(content: str) -> list[dict]:
    """从回复内容中提取文献信息（预留接口，当前返回空列表）。

    完整实现需要依赖 search/ 模块的 Article 数据结构，
    这里仅作占位，便于后续扩展。

    Args:
        content: 助手回复文本。

    Returns:
        文献字典列表。
    """
    return []


def render_chat_area(
    session_id: str,
    user_id: int,
    model_adapter: Optional[Any] = None,
    tool_registry: Optional[dict] = None,
) -> None:
    """渲染主对话区。

    功能：
    - 加载并展示历史消息
    - 提供 st.chat_input 输入框
    - 发送消息 → 调用 LLM → 渲染回复 → 持久化

    Args:
        session_id: 当前会话 ID。
        user_id: 用户 ID。
        model_adapter: 模型适配器（可选，缺失时降级）。
        tool_registry: 工具注册表（可选，缺失时不调用工具）。
    """
    if not session_id:
        st.info("请在左侧选择或创建一个对话。")
        return

    # 加载历史消息（带缓存）
    cache_key = f"chat_messages_{session_id}"
    if cache_key not in st.session_state:
        try:
            st.session_state[cache_key] = get_messages(session_id, limit=100)
        except Exception:
            st.session_state[cache_key] = []

    messages = st.session_state[cache_key]

    # 渲染历史消息
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        msg_id = msg.get("id", 0)
        tool_calls = msg.get("tool_calls")

        if role == "system":
            continue

        if role == "assistant" and tool_calls:
            # 先渲染文本内容
            render_message(role, content, message_id=msg_id)
            # 再渲染工具调用（折叠展示）
            for tc in tool_calls:
                func = tc.get("function", {})
                name = func.get("name", "")
                args_str = func.get("arguments", "{}")
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except (json.JSONDecodeError, TypeError):
                    args = {}
                render_tool_call(tool_name=name, args=args, status="done")
        else:
            render_message(role, content, message_id=msg_id)

    # 输入框
    user_input = st.chat_input("请输入您的问题，例如：最近关于糖尿病与运动的研究有哪些？")

    if user_input and user_input.strip():
        user_text = user_input.strip()

        # 渲染用户消息
        render_message("user", user_text)

        # 持久化用户消息
        try:
            append_message(session_id=session_id, role="user", content=user_text)
        except Exception as e:
            st.error(f"保存消息失败: {e}")
            return

        # 构建上下文
        try:
            history = get_messages(session_id, limit=100)
            context = build_context(history, max_turns=20, max_tokens=8000)
        except Exception as e:
            st.error(f"构建上下文失败: {e}")
            return

        # 生成回复
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("思考中..."):
                reply = _generate_reply(context, model_adapter, tool_registry)

            content = reply.get("content", "")
            tool_calls = reply.get("tool_calls", [])

            # 渲染文本内容
            try:
                st.markdown(content or "(无内容)")
            except Exception:
                st.text(content or "(无内容)")

            # 渲染工具调用（若有）
            for tc in tool_calls:
                func = tc.get("function", {})
                name = func.get("name", "")
                args_str = func.get("arguments", "{}")
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except (json.JSONDecodeError, TypeError):
                    args = {}
                render_tool_call(tool_name=name, args=args, status="done")

            # 提取并渲染文献（预留）
            articles = _extract_articles_from_reply(content)
            if articles:
                render_article_citations(articles)

        # 持久化助手消息
        try:
            append_message(
                session_id=session_id,
                role="assistant",
                content=content,
                tool_calls=tool_calls if tool_calls else None,
            )
        except Exception as e:
            st.error(f"保存回复失败: {e}")

        # 刷新缓存
        try:
            st.session_state[cache_key] = get_messages(session_id, limit=100)
        except Exception:
            pass
