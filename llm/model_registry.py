"""模型注册表 - 定义所有可用 LLM 模型的配置。

每个模型配置包含提供商、API 地址、模型名、API 类型、最大 token 数、
是否支持工具调用、所需用户 tier 等字段。
"""

from typing import Optional

# ---------------------------------------------------------------------------
# 模型注册表（dict 结构，便于按 key 索引）
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, dict] = {
    "deepseek-chat": {
        "provider": "deepseek",
        "display_name": "DeepSeek Chat",
        "base_url": "https://api.deepseek.com",
        "model_name": "deepseek-chat",
        "api_type": "openai_compatible",
        "max_tokens": 8192,
        "supports_tools": True,
        "tier_required": "basic",
    },
    "gpt-4o": {
        "provider": "openai",
        "display_name": "GPT-4o",
        "base_url": "https://api.openai.com/v1",
        "model_name": "gpt-4o",
        "api_type": "openai",
        "max_tokens": 4096,
        "supports_tools": True,
        "tier_required": "pro",
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "display_name": "GPT-4o Mini",
        "base_url": "https://api.openai.com/v1",
        "model_name": "gpt-4o-mini",
        "api_type": "openai",
        "max_tokens": 4096,
        "supports_tools": True,
        "tier_required": "basic",
    },
    "qwen-plus": {
        "provider": "qwen",
        "display_name": "Qwen Plus",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model_name": "qwen-plus",
        "api_type": "openai_compatible",
        "max_tokens": 6000,
        "supports_tools": True,
        "tier_required": "pro",
    },
    "claude-3-5-sonnet": {
        "provider": "anthropic",
        "display_name": "Claude 3.5 Sonnet",
        "base_url": "https://api.anthropic.com",
        "model_name": "claude-3-5-sonnet-20240620",
        "api_type": "anthropic",
        "max_tokens": 8192,
        "supports_tools": True,
        "tier_required": "flagship",
    },
}

# tier 等级顺序（数字越大权限越高）
_TIER_ORDER: dict[str, int] = {
    "free": 0,
    "basic": 1,
    "pro": 2,
    "flagship": 3,
    "institutional": 4,
}


def get_model_config(model_id: str) -> Optional[dict]:
    """根据 model_id 获取模型配置字典，不存在返回 None。"""
    return MODEL_REGISTRY.get(model_id)


def list_models_by_tier(user_tier: str) -> list[str]:
    """按用户 tier 过滤可用模型 ID 列表（用户等级 >= 模型所需等级）。"""
    user_level = _TIER_ORDER.get(user_tier, 0)
    return [
        mid
        for mid, cfg in MODEL_REGISTRY.items()
        if _TIER_ORDER.get(cfg["tier_required"], 0) <= user_level
    ]


def get_tier_level(tier: str) -> int:
    """获取 tier 对应的等级数字（用于比较）。"""
    return _TIER_ORDER.get(tier, 0)
