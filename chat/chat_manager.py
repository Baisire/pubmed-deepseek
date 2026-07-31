"""对话管理器 - 会话生命周期管理。

封装 db.py 中的对话函数，提供统一的会话和消息管理接口。
消息格式兼容 OpenAI Function Calling 格式（role/content/tool_calls/tool_call_id）。
"""

import json
from typing import Optional

import db


def get_or_create_session(user_id: int, session_id: Optional[str] = None) -> str:
    """获取或创建会话。

    若提供了 session_id 且该会话存在，则返回该 session_id；
    否则创建一个新的会话并返回其 ID。

    Args:
        user_id: 用户 ID。
        session_id: 可选的会话 ID；为 None 时强制创建新会话。

    Returns:
        会话 ID 字符串。

    Raises:
        ValueError: user_id 无效时抛出。
    """
    if not user_id or user_id <= 0:
        raise ValueError(f"无效的 user_id: {user_id}")

    if session_id:
        try:
            session = db.get_chat_session(session_id)
            if session and session.get("user_id") == user_id:
                return session_id
        except Exception:
            # 会话不存在或查询失败，走创建分支
            pass

    try:
        new_id = db.create_chat_session(user_id=user_id)
        return new_id
    except Exception as e:
        raise RuntimeError(f"创建会话失败: {e}") from e


def list_sessions(user_id: int) -> list[dict]:
    """获取用户的对话会话列表（按更新时间倒序）。

    Args:
        user_id: 用户 ID。

    Returns:
        会话字典列表，每条包含 id/title/model/created_at/updated_at。
    """
    try:
        return db.list_chat_sessions(user_id=user_id, limit=50)
    except Exception:
        return []


def delete_session(session_id: str) -> None:
    """删除对话会话及其所有消息。

    Args:
        session_id: 会话 ID。

    Raises:
        ValueError: session_id 为空时抛出。
    """
    if not session_id:
        raise ValueError("session_id 不能为空")
    try:
        db.delete_chat_session(session_id)
    except Exception as e:
        raise RuntimeError(f"删除会话失败: {e}") from e


def rename_session(session_id: str, title: str) -> None:
    """重命名对话会话。

    Args:
        session_id: 会话 ID。
        title: 新的会话标题。

    Raises:
        ValueError: session_id 为空或 title 为空时抛出。
    """
    if not session_id:
        raise ValueError("session_id 不能为空")
    if not title or not title.strip():
        raise ValueError("title 不能为空")
    try:
        db.update_chat_session_title(session_id, title.strip())
    except Exception as e:
        raise RuntimeError(f"重命名会话失败: {e}") from e


def append_message(
    session_id: str,
    role: str,
    content: str,
    tool_calls: Optional[list[dict]] = None,
    tool_call_id: Optional[str] = None,
) -> int:
    """追加一条对话消息。

    Args:
        session_id: 会话 ID。
        role: 消息角色（system / user / assistant / tool）。
        content: 消息内容。
        tool_calls: 工具调用列表（仅 assistant 消息使用），
            会被序列化为 JSON 字符串存储。
        tool_call_id: 工具调用 ID（仅 tool 消息使用）。

    Returns:
        消息 ID。

    Raises:
        ValueError: 参数校验失败时抛出。
        RuntimeError: 数据库写入失败时抛出。
    """
    if role not in ("system", "user", "assistant", "tool"):
        raise ValueError(f"无效的 role: {role}")
    if not session_id:
        raise ValueError("session_id 不能为空")

    tool_calls_json: Optional[str] = None
    if tool_calls is not None:
        try:
            tool_calls_json = json.dumps(tool_calls, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            raise ValueError(f"tool_calls 序列化失败: {e}") from e

    try:
        msg_id = db.add_chat_message(
            session_id=session_id,
            role=role,
            content=content,
            tool_calls=tool_calls_json,
            tool_call_id=tool_call_id,
        )
        return msg_id
    except Exception as e:
        raise RuntimeError(f"追加消息失败: {e}") from e


def get_messages(session_id: str, limit: int = 50) -> list[dict]:
    """获取会话的消息列表（按时间正序）。

    tool_calls 字段会从 JSON 字符串反序列化为列表。

    Args:
        session_id: 会话 ID。
        limit: 返回的最大消息数。

    Returns:
        消息字典列表，每条包含 id/role/content/tool_calls/tool_call_id/created_at。
    """
    try:
        rows = db.get_chat_messages(session_id=session_id, limit=limit)
    except Exception:
        return []

    messages: list[dict] = []
    for row in rows:
        msg = dict(row)
        if msg.get("tool_calls"):
            try:
                msg["tool_calls"] = json.loads(msg["tool_calls"])
            except (json.JSONDecodeError, TypeError):
                msg["tool_calls"] = None
        else:
            msg["tool_calls"] = None
        messages.append(msg)
    return messages
