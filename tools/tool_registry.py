"""工具注册表 - 注册所有 AI 工具并提供 OpenAI Function Calling 格式转换。"""

from typing import Any

from . import pubmed_search_tool
from . import literature_analysis_tool
from . import mesh_lookup_tool


# 工具名 -> 模块 映射
_TOOL_MODULES: dict[str, Any] = {
    pubmed_search_tool.TOOL_NAME: pubmed_search_tool,
    literature_analysis_tool.TOOL_NAME: literature_analysis_tool,
    mesh_lookup_tool.TOOL_NAME: mesh_lookup_tool,
}


# 工具元数据：名称、描述、参数 schema、执行函数
TOOL_DEFINITIONS: dict[str, dict[str, Any]] = {
    name: {
        "name": name,
        "description": module.TOOL_DESCRIPTION,
        "parameters": module.TOOL_PARAMETERS,
        "execute": module.execute,
    }
    for name, module in _TOOL_MODULES.items()
}


def to_openai_tools(tool_names: list[str]) -> list[dict[str, Any]]:
    """将指定工具列表转换为 OpenAI Function Calling 格式。

    Args:
        tool_names: 需要转换的工具名列表

    Returns:
        OpenAI tools 格式的字典列表，每项为
        {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}

    Raises:
        ValueError: 当 tool_names 中包含未注册的工具名
    """
    tools: list[dict[str, Any]] = []
    for name in tool_names:
        if name not in TOOL_DEFINITIONS:
            raise ValueError(f"未注册的工具：{name}，可用工具：{list(TOOL_DEFINITIONS.keys())}")
        definition = TOOL_DEFINITIONS[name]
        tools.append({
            "type": "function",
            "function": {
                "name": definition["name"],
                "description": definition["description"],
                "parameters": definition["parameters"],
            },
        })
    return tools


def execute_tool(tool_name: str,
                 args: dict[str, Any],
                 context: dict[str, Any] | None = None) -> dict[str, Any]:
    """执行指定工具并返回结果。

    Args:
        tool_name: 工具名
        args: 工具参数字典（由 LLM 解析得到）
        context: 调用上下文，含 user_id、api_key 等环境信息

    Returns:
        工具执行结果字典；工具不存在或执行失败时返回 {"error": "原因"}
    """
    if tool_name not in TOOL_DEFINITIONS:
        return {"error": f"未注册的工具：{tool_name}"}

    context = context or {}
    execute_fn = TOOL_DEFINITIONS[tool_name]["execute"]

    try:
        return execute_fn(**args, context=context)
    except TypeError as e:
        return {"error": f"工具参数错误：{e}"}
    except Exception as e:
        return {"error": f"工具执行异常：{e}"}
