"""上下文管理器 - 构建发送给 LLM 的消息列表。

负责将系统提示词、对话历史、工具调用结果整合成符合
OpenAI Function Calling 格式的消息列表，供 LLM 推理使用。

核心特性：
- 对话历史最多保留 N 轮（user/assistant 为一轮，tool 消息跟随 assistant）
- 工具调用结果过长时自动摘要或截断
- 系统提示词始终保留在顶部
- 输出为 OpenAI 兼容的 messages 格式
"""

import re
from typing import Callable, Optional

from chat.chat_manager import get_messages

DEFAULT_SYSTEM_PROMPT: str = (
    "你是一位医学文献智能助手，擅长 PubMed 检索和文献解读。\n"
    "你的回答必须专业、准确、有依据，所有医学结论都应标注文献来源（PMID）。\n"
    "当用户需要检索最新研究、分析文献、翻译摘要或对比不同研究时，\n"
    "你应调用 PubMed 检索工具获取文献，再基于检索结果给出回答。\n"
    "回答时注意：\n"
    "1. 区分证据等级，明确哪些是结论性证据，哪些是初步发现；\n"
    "2. 不编造文献，不确定的内容坦诚说明；\n"
    "3. 引用文献时使用 [PMID:xxxx] 格式标注；\n"
    "4. 对于临床相关问题，提醒用户咨询专业医师，不提供诊疗建议。"
)

# 工具结果摘要阈值（字符数）
TOOL_RESULT_SUMMARY_THRESHOLD: int = 4000


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数。

    估算公式：中文字符数 + 英文词数 * 1.3，取整。
    适用于快速判断上下文长度，不要求精确。

    Args:
        text: 待估算的文本。

    Returns:
        估算的 token 数。
    """
    if not text:
        return 0
    # 中文字符（含中文标点）
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]", text))
    # 英文单词（连续的英文字母/数字串）
    english_words = len(re.findall(r"[A-Za-z0-9]+", text))
    return int(chinese_chars + english_words * 1.3)


def _truncate_by_turns(messages: list[dict], max_turns: int) -> list[dict]:
    """按轮次截断消息列表。

    一轮对话定义为：一条 user 消息 + 对应的 assistant 消息及其 tool 回复。
    从尾部向前数 max_turns 轮，保留这些轮的所有消息。

    Args:
        messages: 消息列表（按时间正序）。
        max_turns: 最大保留轮数。

    Returns:
        截断后的消息列表（保持正序）。
    """
    if max_turns <= 0:
        return []
    if len(messages) <= max_turns:
        return list(messages)

    user_positions: list[int] = [
        i for i, m in enumerate(messages) if m.get("role") == "user"
    ]

    if len(user_positions) <= max_turns:
        return list(messages)

    start_idx = user_positions[-max_turns]
    return list(messages[start_idx:])


def _summarize_tool_result(
    content: str,
    summarizer: Optional[Callable[[str], str]],
) -> str:
    """对过长的工具结果进行摘要或截断。

    Args:
        content: 原始工具结果文本。
        summarizer: 摘要函数，接受文本返回摘要；
            为 None 时直接截断并添加提示。

    Returns:
        摘要或截断后的文本。
    """
    if len(content) <= TOOL_RESULT_SUMMARY_THRESHOLD:
        return content

    if summarizer is not None:
        try:
            prompt = (
                "请用中文简洁总结以下 PubMed 检索结果，"
                "保留关键文献信息（标题、PMID、核心结论）：\n\n"
                f"{content[:8000]}"
            )
            summary = summarizer(prompt)
            if summary and len(summary) < len(content):
                return (
                    f"【检索结果已自动摘要（原文约 {len(content)} 字）】\n"
                    f"{summary}"
                )
        except Exception:
            pass

    # 降级：直接截断
    truncated = content[: TOOL_RESULT_SUMMARY_THRESHOLD - 50]
    return (
        f"【检索结果过长已截断，仅展示前 {TOOL_RESULT_SUMMARY_THRESHOLD - 50} 字】\n"
        f"{truncated}\n……"
    )


def build_context(
    messages: list[dict],
    max_turns: int = 20,
    max_tokens: int = 8000,
    system_prompt: Optional[str] = None,
    summarizer: Optional[Callable[[str], str]] = None,
) -> list[dict]:
    """构建 LLM 可用的上下文消息列表（OpenAI 格式）。

    组成结构：
    1. 系统提示词（system role，始终保留在顶部）
    2. 最近 N 轮对话历史（已截断）
    3. 对过长的 tool 消息进行摘要或截断
    4. 若总 token 仍超 max_tokens，进一步从历史头部删除轮次

    Args:
        messages: 原始消息列表（按时间正序），格式与 get_messages 返回一致。
        max_turns: 最大保留对话轮数，默认 20。
        max_tokens: 最大 token 数（粗略估算），默认 8000。
        system_prompt: 自定义系统提示词，为 None 时使用默认医学提示词。
        summarizer: 长工具结果摘要函数，为 None 时使用截断降级。

    Returns:
        符合 OpenAI 格式的消息字典列表。
    """
    sys_prompt = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT

    # 先按轮次做基础截断
    truncated = _truncate_by_turns(messages, max_turns=max_turns)

    # 格式化并可能摘要工具结果
    formatted: list[dict] = []
    for msg in truncated:
        role = msg.get("role", "user")
        content = msg.get("content", "") or ""

        if role == "tool":
            content = _summarize_tool_result(content, summarizer)

        out_msg: dict = {"role": role, "content": content}

        tool_calls = msg.get("tool_calls")
        if tool_calls and role == "assistant":
            out_msg["tool_calls"] = tool_calls

        tool_call_id = msg.get("tool_call_id")
        if tool_call_id and role == "tool":
            out_msg["tool_call_id"] = tool_call_id

        formatted.append(out_msg)

    # 从头部逐步删除轮次，直到低于 max_tokens（保留 system 提示词）
    result: list[dict] = [{"role": "system", "content": sys_prompt}] + formatted

    try:
        total_tokens = estimate_tokens(sys_prompt) + sum(
            estimate_tokens(str(m.get("content", ""))) for m in formatted
        )
    except Exception:
        total_tokens = 0

    while total_tokens > max_tokens and len(formatted) > 0:
        # 从 formatted 头部移除一轮（找到第一条 user 消息及其后到下一条 user 之前的消息）
        first_user_idx = None
        for i, m in enumerate(formatted):
            if m.get("role") == "user":
                first_user_idx = i
                break
        if first_user_idx is None:
            break

        # 找到第二条 user 消息的位置
        second_user_idx = None
        for i in range(first_user_idx + 1, len(formatted)):
            if formatted[i].get("role") == "user":
                second_user_idx = i
                break

        # 删除第一条 user 及其后直到下一条 user 之前的消息
        end_idx = second_user_idx if second_user_idx is not None else len(formatted)
        removed = formatted[first_user_idx:end_idx]
        formatted = formatted[:first_user_idx] + formatted[end_idx:]

        removed_tokens = sum(
            estimate_tokens(str(m.get("content", ""))) for m in removed
        )
        total_tokens -= removed_tokens

        result = [{"role": "system", "content": sys_prompt}] + formatted

    return result
