"""商业化分层 - Tier 权限与用量限额。

定义各 tier 的功能权限和用量限制，提供统一的 gating 检查函数。
"""

import db

# 各 tier 的功能权限配置
TIER_FEATURES: dict[str, dict] = {
    db.TIER_FREE: {
        "name": "免费版",
        "daily_search_limit": 5,           # 每日检索次数
        "max_articles_per_search": 5,      # 每次检索文献数
        "semantic_rerank": False,          # 语义精排
        "cross_encoder": False,            # Cross-Encoder 精排
        "citation_boost": False,           # 引文排序
        "chat_mode": False,                # 对话模式
        "tool_calling": False,             # 工具调用
        "available_models": ["deepseek-chat"],  # 空=对话不可用
    },
    db.TIER_BASIC: {
        "name": "基础版",
        "daily_search_limit": 20,
        "max_articles_per_search": 20,
        "semantic_rerank": True,
        "cross_encoder": False,
        "citation_boost": False,
        "chat_mode": True,
        "tool_calling": True,
        "available_models": ["deepseek-chat", "gpt-4o-mini"],
    },
    db.TIER_PRO: {
        "name": "专业版",
        "daily_search_limit": 50,
        "max_articles_per_search": 50,
        "semantic_rerank": True,
        "cross_encoder": True,
        "citation_boost": True,
        "chat_mode": True,
        "tool_calling": True,
        "available_models": [
            "deepseek-chat", "gpt-4o-mini", "gpt-4o", "qwen-plus",
        ],
    },
    db.TIER_FLAGSHIP: {
        "name": "旗舰版",
        "daily_search_limit": 200,
        "max_articles_per_search": 100,
        "semantic_rerank": True,
        "cross_encoder": True,
        "citation_boost": True,
        "chat_mode": True,
        "tool_calling": True,
        "available_models": [
            "deepseek-chat", "gpt-4o-mini", "gpt-4o",
            "qwen-plus", "claude-3-5-sonnet",
        ],
    },
    db.TIER_INSTITUTIONAL: {
        "name": "机构版",
        "daily_search_limit": 1000,
        "max_articles_per_search": 200,
        "semantic_rerank": True,
        "cross_encoder": True,
        "citation_boost": True,
        "chat_mode": True,
        "tool_calling": True,
        "available_models": [
            "deepseek-chat", "gpt-4o-mini", "gpt-4o",
            "qwen-plus", "claude-3-5-sonnet",
        ],
    },
}


def get_tier_features(tier: str) -> dict:
    """获取指定 tier 的功能配置。"""
    return TIER_FEATURES.get(tier, TIER_FEATURES[db.TIER_FREE])


def can_use_feature(tier: str, feature: str) -> bool:
    """检查 tier 是否有权使用指定功能。

    feature 可选值: semantic_rerank / cross_encoder / citation_boost /
                   chat_mode / tool_calling
    """
    features = get_tier_features(tier)
    return features.get(feature, False)


def get_daily_limit(tier: str) -> int:
    """获取每日检索次数限额。"""
    return get_tier_features(tier).get("daily_search_limit", 5)


def get_max_articles(tier: str) -> int:
    """获取每次检索最大文献数。"""
    return get_tier_features(tier).get("max_articles_per_search", 5)


def check_daily_quota(user_id: int) -> tuple[bool, int, int]:
    """检查用户今日用量是否还有剩余。

    Returns:
        (是否可用, 已用次数, 总限额)
    """
    tier = db.get_user_tier(user_id)
    limit = get_daily_limit(tier)
    used = db.get_daily_usage_count(user_id)
    return used < limit, used, limit


def clamp_max_results(user_tier: str, requested: int) -> int:
    """根据 tier 限制实际返回的文献数。"""
    limit = get_max_articles(user_tier)
    return min(requested, limit)


def get_tier_display_name(tier: str) -> str:
    """获取 tier 的中文显示名。"""
    return get_tier_features(tier).get("name", tier)
