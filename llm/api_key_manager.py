"""API Key 管理器 - 用户级 Key 存储与环境变量兜底。

优先级：用户存储 (db) > 环境变量 > None

提供商与环境变量映射：
- deepseek    -> DEEPSEEK_API_KEY
- openai      -> OPENAI_API_KEY
- qwen        -> DASHSCOPE_API_KEY
- anthropic   -> ANTHROPIC_API_KEY
"""

import os
from typing import Optional

from .model_registry import MODEL_REGISTRY


# 提供商 -> 环境变量名映射
PROVIDER_ENV_MAP: dict[str, str] = {
    "deepseek": "DEEPSEEK_API_KEY",
    "openai": "OPENAI_API_KEY",
    "qwen": "DASHSCOPE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


# ---------------------------------------------------------------------------
# 基础操作
# ---------------------------------------------------------------------------

def get_api_key(user_id: int, provider: str) -> Optional[str]:
    """从数据库获取用户指定提供商的 API Key。

    未设置或读取失败返回 None。

    Args:
        user_id: 用户 ID
        provider: 提供商名称（deepseek / openai / qwen / anthropic）

    Returns:
        解密后的 API Key 字符串，未找到返回 None
    """
    try:
        from db import get_user_api_key  # 延迟导入避免循环依赖

        return get_user_api_key(user_id, provider)
    except Exception:
        return None


def set_api_key(user_id: int, provider: str, key: str) -> None:
    """保存用户的 API Key（加密存储到数据库）。

    Args:
        user_id: 用户 ID
        provider: 提供商名称
        key: API Key 明文
    """
    try:
        from db import set_user_api_key  # 延迟导入避免循环依赖

        set_user_api_key(user_id, provider, key)
    except Exception as exc:
        raise RuntimeError(f"保存 API Key 失败 (provider: {provider}): {exc}") from exc


# ---------------------------------------------------------------------------
# 模型 → provider 映射
# ---------------------------------------------------------------------------

def get_provider_for_model(model_id: str) -> str:
    """获取模型对应的 provider 名称。

    Args:
        model_id: 模型 ID

    Returns:
        provider 名称

    Raises:
        KeyError: 模型未注册
    """
    cfg = MODEL_REGISTRY.get(model_id)
    if cfg is None:
        raise KeyError(f"未知模型: {model_id}")
    return cfg["provider"]


# ---------------------------------------------------------------------------
# 综合解析：用户配置 → 环境变量兜底
# ---------------------------------------------------------------------------

def resolve_api_key(
    model_id: str,
    user_id: int,
    env_fallback: bool = True,
) -> Optional[str]:
    """解析模型对应的 API Key。

    优先级：用户数据库存储 > 环境变量（env_fallback=True 时）

    Args:
        model_id: 模型 ID
        user_id: 用户 ID
        env_fallback: 是否允许环境变量兜底，默认 True

    Returns:
        API Key 字符串，未找到返回 None
    """
    provider = get_provider_for_model(model_id)

    # 1. 优先从用户存储获取
    user_key = get_api_key(user_id, provider)
    if user_key:
        return user_key

    # 2. 环境变量兜底
    if env_fallback:
        env_name = PROVIDER_ENV_MAP.get(provider, "")
        if env_name:
            env_key = os.environ.get(env_name)
            if env_key:
                return env_key

    return None
