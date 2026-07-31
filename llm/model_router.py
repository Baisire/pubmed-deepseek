"""模型路由器 - 模型选择、可用性过滤与降级链。

功能：
- 根据用户 tier 返回可用模型列表
- 获取模型配置
- 构建降级链（主模型 → 同 provider 备用 → 其他 provider 同级别 → 基础模型）
- 带降级的聊天调用（主模型失败后依次尝试降级链）
"""

from typing import Optional

from .model_registry import MODEL_REGISTRY, get_tier_level
from .model_adapter import LLMResponse, LLMError, chat_completion


# ---------------------------------------------------------------------------
# 基础路由功能
# ---------------------------------------------------------------------------

def get_available_models(user_tier: str) -> list[str]:
    """根据用户 tier 返回可用模型 ID 列表（按性价比从高到低排序）。

    Args:
        user_tier: 用户 tier 等级（free / basic / pro / flagship / institutional）

    Returns:
        可用的模型 ID 列表
    """
    user_level = get_tier_level(user_tier)
    # 推荐排序（性价比从高到低）
    priority_order = [
        "deepseek-chat",
        "gpt-4o-mini",
        "qwen-plus",
        "gpt-4o",
        "claude-3-5-sonnet",
    ]
    return [
        mid
        for mid in priority_order
        if mid in MODEL_REGISTRY
        and get_tier_level(MODEL_REGISTRY[mid]["tier_required"]) <= user_level
    ]


def get_model_config(model_id: str) -> dict:
    """获取模型配置字典。

    Args:
        model_id: 模型 ID

    Returns:
        模型配置字典

    Raises:
        KeyError: 模型不存在
    """
    cfg = MODEL_REGISTRY.get(model_id)
    if cfg is None:
        raise KeyError(f"未知模型: {model_id}")
    return cfg


# ---------------------------------------------------------------------------
# 降级链构建
# ---------------------------------------------------------------------------

def build_degradation_chain(preferred_model: str, user_tier: str) -> list[str]:
    """构建降级链。

    降级顺序：
    1. 同 provider 的低 tier 模型
    2. 其他 provider 同级别模型
    3. 更低 tier 的基础模型
    所有候选均受 user_tier 约束（必须在用户权限范围内）。

    Args:
        preferred_model: 首选模型 ID
        user_tier: 用户 tier 等级

    Returns:
        降级模型 ID 列表（不含首选模型本身）
    """
    if preferred_model not in MODEL_REGISTRY:
        return []

    preferred_cfg = MODEL_REGISTRY[preferred_model]
    preferred_provider = preferred_cfg["provider"]
    preferred_tier = preferred_cfg["tier_required"]
    preferred_tier_level = get_tier_level(preferred_tier)
    user_level = get_tier_level(user_tier)

    same_provider_other: list[str] = []      # 同 provider 其他模型（tier <= 首选）
    same_tier_other: list[str] = []          # 其他 provider 同 tier 模型
    lower_tier_models: list[str] = []        # 更低 tier 的模型

    # 性价比参考排序
    cheap_first = [
        "deepseek-chat",
        "gpt-4o-mini",
        "qwen-plus",
        "gpt-4o",
        "claude-3-5-sonnet",
    ]

    for mid in cheap_first:
        if mid == preferred_model:
            continue
        cfg = MODEL_REGISTRY.get(mid)
        if cfg is None:
            continue
        # 超出用户权限的跳过
        if get_tier_level(cfg["tier_required"]) > user_level:
            continue

        model_tier_level = get_tier_level(cfg["tier_required"])

        if cfg["provider"] == preferred_provider and model_tier_level <= preferred_tier_level:
            same_provider_other.append(mid)
        elif model_tier_level == preferred_tier_level:
            same_tier_other.append(mid)
        elif model_tier_level < preferred_tier_level:
            lower_tier_models.append(mid)

    chain = same_provider_other + same_tier_other + lower_tier_models

    # 去重（保持顺序）
    seen: set[str] = set()
    result: list[str] = []
    for mid in chain:
        if mid not in seen:
            seen.add(mid)
            result.append(mid)
    return result


# ---------------------------------------------------------------------------
# 带降级的聊天调用
# ---------------------------------------------------------------------------

def chat_with_fallback(
    model_id: str,
    messages: list[dict],
    api_key_map: dict[str, str],
    tools: Optional[list[dict]] = None,
    **kwargs,
) -> LLMResponse:
    """带降级策略的聊天调用。

    主模型失败时，依次尝试降级链中的其他模型，直到成功或全部失败。

    Args:
        model_id: 首选模型 ID
        messages: 消息列表（OpenAI 格式）
        api_key_map: {provider: api_key} 字典，每个提供商对应一个 key
        tools: 工具定义（OpenAI Function Calling 格式）
        **kwargs: 其他参数（temperature、max_tokens 等）传递给 chat_completion

    Returns:
        第一个成功的 LLMResponse

    Raises:
        LLMError: 所有模型均调用失败
    """
    errors: list[str] = []

    # 确定用户 tier（从主模型推断，用于限制降级链范围）
    main_cfg = get_model_config(model_id)
    user_tier = _infer_user_tier(api_key_map)
    chain = [model_id] + build_degradation_chain(model_id, user_tier)

    for mid in chain:
        cfg = get_model_config(mid)
        provider = cfg["provider"]
        api_key = api_key_map.get(provider, "")
        if not api_key:
            errors.append(f"[{mid}] 无可用 API Key (provider: {provider})")
            continue

        try:
            return chat_completion(
                model_id=mid,
                messages=messages,
                api_key=api_key,
                tools=tools,
                **kwargs,
            )
        except LLMError as exc:
            errors.append(f"[{mid}] {exc}")
            continue

    raise LLMError(
        "所有模型均调用失败:\n" + "\n".join(f"  - {e}" for e in errors)
    )


def _infer_user_tier(api_key_map: dict[str, str]) -> str:
    """根据可用的 API Key 推断用户最高可用 tier。

    规则：只要某个 provider 有 key，就认为该 provider 下所有模型都可用。
    取所有可用模型中的最高 tier 作为用户 tier。
    """
    max_tier_level = 0
    for mid, cfg in MODEL_REGISTRY.items():
        if api_key_map.get(cfg["provider"], ""):
            level = get_tier_level(cfg["tier_required"])
            if level > max_tier_level:
                max_tier_level = level
    # 反向映射 tier 名称
    for tier_name, tier_level in [
        ("free", 0), ("basic", 1), ("pro", 2), ("flagship", 3), ("institutional", 4)
    ]:
        if tier_level == max_tier_level:
            return tier_name
    return "basic"
