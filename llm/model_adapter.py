"""模型适配器 - 统一不同 LLM API 的调用接口。

屏蔽 OpenAI / Anthropic 等不同提供商的 API 差异，对外提供一致的
chat_completion 与 stream_chat_completion 接口。

设计要点：
- 客户端懒加载（首次调用时创建并缓存）
- 工具调用格式统一为 OpenAI Function Calling 格式
- 任何失败抛出自定义 LLMError 异常
"""

import json
from dataclasses import dataclass, field
from typing import Generator, Optional

from .model_registry import get_model_config


# ---------------------------------------------------------------------------
# 自定义异常
# ---------------------------------------------------------------------------

class LLMError(Exception):
    """LLM 调用失败时抛出的统一异常。"""
    pass


# ---------------------------------------------------------------------------
# 统一响应数据结构
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    """统一的 LLM 响应格式。"""

    content: str                           # 文本回复内容
    tool_calls: list[dict] = field(default_factory=list)   # OpenAI 格式工具调用
    usage: dict = field(default_factory=dict)               # token 用量统计
    model: str = ""                        # 实际使用的模型 ID


# ---------------------------------------------------------------------------
# 客户端缓存（懒加载）
# ---------------------------------------------------------------------------

_openai_clients: dict[tuple[str, str], object] = {}   # (base_url, api_key) -> client
_anthropic_client: Optional[object] = None
_anthropic_available: bool = True


def _get_openai_client(base_url: str, api_key: str) -> object:
    """获取或创建 OpenAI 兼容客户端（懒加载 + 缓存）。"""
    cache_key = (base_url, api_key)
    if cache_key not in _openai_clients:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMError("openai 库未安装，请先安装 openai 依赖") from exc
        _openai_clients[cache_key] = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
    return _openai_clients[cache_key]


def _get_anthropic_client(api_key: str) -> object:
    """获取或创建 Anthropic 客户端（懒加载 + 缓存）。"""
    global _anthropic_client, _anthropic_available
    if not _anthropic_available:
        raise LLMError("anthropic 库未安装，请先安装 anthropic 依赖以使用 Claude 模型")
    if _anthropic_client is not None:
        return _anthropic_client
    try:
        from anthropic import Anthropic
        _anthropic_client = Anthropic(api_key=api_key)
    except ImportError as exc:
        _anthropic_available = False
        raise LLMError(
            "anthropic 库未安装，请先安装 anthropic 依赖以使用 Claude 模型"
        ) from exc
    return _anthropic_client


# ---------------------------------------------------------------------------
# 工具格式转换（OpenAI ↔ Anthropic）
# ---------------------------------------------------------------------------

def _convert_tools_to_anthropic(tools: list[dict]) -> list[dict]:
    """将 OpenAI 工具格式转换为 Anthropic 工具格式。

    OpenAI: {"type": "function", "function": {"name", "description", "parameters"}}
    Anthropic: {"name", "description", "input_schema"}
    """
    result: list[dict] = []
    for tool in tools:
        if tool.get("type") == "function" and "function" in tool:
            fn = tool["function"]
            result.append({
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            })
    return result


def _convert_tool_calls_to_openai(anthropic_tool_calls: list) -> list[dict]:
    """将 Anthropic 工具调用转换为 OpenAI 格式。"""
    result: list[dict] = []
    for idx, tc in enumerate(anthropic_tool_calls):
        if hasattr(tc, "name"):
            name = tc.name
            input_data = tc.input if hasattr(tc, "input") else {}
            arguments_str = json.dumps(input_data, ensure_ascii=False)
            result.append({
                "id": getattr(tc, "id", f"toolu_{idx}"),
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments_str,
                },
            })
        elif isinstance(tc, dict):
            name = tc.get("name", "")
            input_data = tc.get("input", {})
            arguments_str = json.dumps(input_data, ensure_ascii=False)
            result.append({
                "id": tc.get("id", f"toolu_{idx}"),
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments_str,
                },
            })
    return result


def _extract_system_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    """从消息列表中提取 system 消息，返回 (system_str, 剩余消息)。"""
    system_parts: list[str] = []
    remaining: list[dict] = []
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        system_parts.append(block.get("text", ""))
        else:
            remaining.append(msg)
    return "\n\n".join(system_parts), remaining


# ---------------------------------------------------------------------------
# OpenAI 兼容调用
# ---------------------------------------------------------------------------

def _chat_openai_compatible(
    model_id: str,
    messages: list[dict],
    api_key: str,
    tools: Optional[list[dict]] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> LLMResponse:
    """使用 OpenAI 兼容 SDK 调用（deepseek / openai / qwen 都走这里）。"""
    cfg = get_model_config(model_id)
    if cfg is None:
        raise LLMError(f"未知模型: {model_id}")

    client = _get_openai_client(cfg["base_url"], api_key)

    effective_max_tokens = max_tokens if max_tokens is not None else cfg["max_tokens"]

    kwargs: dict = {
        "model": cfg["model_name"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": effective_max_tokens,
    }
    if tools is not None:
        kwargs["tools"] = tools

    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as exc:
        raise LLMError(f"OpenAI 兼容调用失败 ({model_id}): {exc}") from exc

    choice = response.choices[0]
    message = choice.message

    tool_calls: list[dict] = []
    if hasattr(message, "tool_calls") and message.tool_calls:
        for tc in message.tool_calls:
            tool_calls.append({
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            })

    usage_dict: dict = {}
    if hasattr(response, "usage") and response.usage is not None:
        u = response.usage
        usage_dict = {
            "prompt_tokens": getattr(u, "prompt_tokens", 0),
            "completion_tokens": getattr(u, "completion_tokens", 0),
            "total_tokens": getattr(u, "total_tokens", 0),
        }

    return LLMResponse(
        content=message.content or "",
        tool_calls=tool_calls,
        usage=usage_dict,
        model=model_id,
    )


def _stream_openai_compatible(
    model_id: str,
    messages: list[dict],
    api_key: str,
    tools: Optional[list[dict]] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> Generator[str, None, None]:
    """OpenAI 兼容 SDK 的流式调用生成器。"""
    cfg = get_model_config(model_id)
    if cfg is None:
        raise LLMError(f"未知模型: {model_id}")

    client = _get_openai_client(cfg["base_url"], api_key)

    effective_max_tokens = max_tokens if max_tokens is not None else cfg["max_tokens"]

    kwargs: dict = {
        "model": cfg["model_name"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": effective_max_tokens,
        "stream": True,
    }
    if tools is not None:
        kwargs["tools"] = tools

    try:
        stream = client.chat.completions.create(**kwargs)
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as exc:
        raise LLMError(f"OpenAI 兼容流式调用失败 ({model_id}): {exc}") from exc


# ---------------------------------------------------------------------------
# Anthropic 调用
# ---------------------------------------------------------------------------

def _chat_anthropic(
    model_id: str,
    messages: list[dict],
    api_key: str,
    tools: Optional[list[dict]] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> LLMResponse:
    """使用 Anthropic SDK 调用。"""
    cfg = get_model_config(model_id)
    if cfg is None:
        raise LLMError(f"未知模型: {model_id}")

    client = _get_anthropic_client(api_key)

    effective_max_tokens = max_tokens if max_tokens is not None else cfg["max_tokens"]

    # 1. 提取 system 消息（Anthropic 要求 system 单独传参）
    system_prompt, non_system_messages = _extract_system_messages(messages)

    # 2. 工具格式转换
    anthropic_tools = _convert_tools_to_anthropic(tools) if tools else None

    kwargs: dict = {
        "model": cfg["model_name"],
        "messages": non_system_messages,
        "temperature": temperature,
        "max_tokens": effective_max_tokens,
    }
    if system_prompt:
        kwargs["system"] = system_prompt
    if anthropic_tools:
        kwargs["tools"] = anthropic_tools

    try:
        response = client.messages.create(**kwargs)
    except Exception as exc:
        raise LLMError(f"Anthropic 调用失败 ({model_id}): {exc}") from exc

    # 3. 解析响应
    content_text: str = ""
    tool_calls_raw: list = []

    content_blocks = getattr(response, "content", [])
    if isinstance(content_blocks, list):
        for block in content_blocks:
            block_type = (
                getattr(block, "type", None)
                if not isinstance(block, dict)
                else block.get("type")
            )
            if block_type == "text":
                text = (
                    getattr(block, "text", "")
                    if not isinstance(block, dict)
                    else block.get("text", "")
                )
                content_text += text
            elif block_type == "tool_use":
                tool_calls_raw.append(block)

    # 4. 用量统计
    usage_dict: dict = {}
    usage_obj = getattr(response, "usage", None)
    if usage_obj is not None:
        input_tokens = getattr(usage_obj, "input_tokens", 0)
        output_tokens = getattr(usage_obj, "output_tokens", 0)
        usage_dict = {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

    return LLMResponse(
        content=content_text,
        tool_calls=_convert_tool_calls_to_openai(tool_calls_raw),
        usage=usage_dict,
        model=model_id,
    )


def _stream_anthropic(
    model_id: str,
    messages: list[dict],
    api_key: str,
    tools: Optional[list[dict]] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> Generator[str, None, None]:
    """Anthropic SDK 的流式调用生成器。"""
    cfg = get_model_config(model_id)
    if cfg is None:
        raise LLMError(f"未知模型: {model_id}")

    client = _get_anthropic_client(api_key)

    effective_max_tokens = max_tokens if max_tokens is not None else cfg["max_tokens"]

    system_prompt, non_system_messages = _extract_system_messages(messages)
    anthropic_tools = _convert_tools_to_anthropic(tools) if tools else None

    kwargs: dict = {
        "model": cfg["model_name"],
        "messages": non_system_messages,
        "temperature": temperature,
        "max_tokens": effective_max_tokens,
    }
    if system_prompt:
        kwargs["system"] = system_prompt
    if anthropic_tools:
        kwargs["tools"] = anthropic_tools

    try:
        with client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield text
    except Exception as exc:
        raise LLMError(f"Anthropic 流式调用失败 ({model_id}): {exc}") from exc


# ---------------------------------------------------------------------------
# 统一对外接口
# ---------------------------------------------------------------------------

def chat_completion(
    model_id: str,
    messages: list[dict],
    api_key: str,
    tools: Optional[list[dict]] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> LLMResponse:
    """统一的 LLM 对话接口（非流式）。

    根据 model_id 自动选择对应的提供商 SDK 并调用。

    Args:
        model_id: 模型 ID，必须在 MODEL_REGISTRY 中注册
        messages: 消息列表（OpenAI 格式，含 role/content）
        api_key: API Key
        tools: 工具定义（OpenAI Function Calling 格式）
        temperature: 温度参数，默认 0.7
        max_tokens: 最大输出 token 数，None 则使用模型默认值

    Returns:
        统一格式的 LLMResponse

    Raises:
        LLMError: 模型不存在、API Key 为空或调用失败
    """
    cfg = get_model_config(model_id)
    if cfg is None:
        raise LLMError(f"未知模型: {model_id}")

    if not api_key:
        raise LLMError(f"API Key 不能为空 (provider: {cfg['provider']})")

    if cfg["api_type"] in ("openai", "openai_compatible"):
        return _chat_openai_compatible(
            model_id=model_id,
            messages=messages,
            api_key=api_key,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif cfg["api_type"] == "anthropic":
        return _chat_anthropic(
            model_id=model_id,
            messages=messages,
            api_key=api_key,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        raise LLMError(f"不支持的 API 类型: {cfg['api_type']}")


def stream_chat_completion(
    model_id: str,
    messages: list[dict],
    api_key: str,
    tools: Optional[list[dict]] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> Generator[str, None, None]:
    """统一的 LLM 流式对话接口（生成器，yield 文本片段）。

    Args:
        model_id: 模型 ID
        messages: 消息列表（OpenAI 格式）
        api_key: API Key
        tools: 工具定义（OpenAI Function Calling 格式）
        temperature: 温度参数
        max_tokens: 最大输出 token 数

    Yields:
        文本片段字符串

    Raises:
        LLMError: 模型不存在或调用失败
    """
    cfg = get_model_config(model_id)
    if cfg is None:
        raise LLMError(f"未知模型: {model_id}")

    if not api_key:
        raise LLMError(f"API Key 不能为空 (provider: {cfg['provider']})")

    if cfg["api_type"] in ("openai", "openai_compatible"):
        yield from _stream_openai_compatible(
            model_id=model_id,
            messages=messages,
            api_key=api_key,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif cfg["api_type"] == "anthropic":
        yield from _stream_anthropic(
            model_id=model_id,
            messages=messages,
            api_key=api_key,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        raise LLMError(f"不支持的 API 类型: {cfg['api_type']}")
