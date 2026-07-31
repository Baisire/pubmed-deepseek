"""多模型调度模块 - 统一接口调用不同 LLM 提供商。

包含：
- model_registry: 模型注册表（5 个模型配置与可用性过滤）
- model_adapter: 模型适配器（统一调用，屏蔽 OpenAI / Anthropic 差异）
- model_router: 模型路由器（可用模型筛选、降级链、fallback 调用）
- api_key_manager: API Key 管理（用户级 / 环境变量级）
"""

from .model_registry import MODEL_REGISTRY, get_model_config
from .model_adapter import LLMResponse, LLMError, chat_completion, stream_chat_completion
from .model_router import (
    get_available_models,
    build_degradation_chain,
    chat_with_fallback,
)
from .api_key_manager import (
    get_api_key,
    set_api_key,
    get_provider_for_model,
    resolve_api_key,
)

__all__ = [
    "MODEL_REGISTRY",
    "get_model_config",
    "LLMResponse",
    "LLMError",
    "chat_completion",
    "stream_chat_completion",
    "get_available_models",
    "build_degradation_chain",
    "chat_with_fallback",
    "get_api_key",
    "set_api_key",
    "get_provider_for_model",
    "resolve_api_key",
]
