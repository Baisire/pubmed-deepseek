# 医学 PubMed 检索 - DeepSeek 文献解读 - Streamlit 应用宪法

## 项目定位
一款基于 Python + Streamlit 的医学文献自动化检索与 AI 解读工具。用户输入医学研究关键词，系统自动调用 PubMed API 抓取对应权威文献，将论文摘要传给 DeepSeek 大模型，AI 一键完成翻译、总结、观点提炼、综述整理，一站式完成科研文献调研。

> 项目状态：**初版完成**。app.py、db.py、提示词、文档、使用说明书已就绪，可直接运行。

### 两个核心工具
1. **PubMed**：全球权威免费医学/生物学术文献数据库，存储海量医学论文、临床研究文献，提供 E-utilities API 检索接口
2. **DeepSeek**：国产专业大模型，擅长学术、科研文本分析，可读文献、翻译英文论文、梳理实验结论、整理文献综述

### 方案说明
本方案核心功能：
- PubMed 检索逻辑：esearch + efetch + XML 解析
- DeepSeek 提示词：三套 System Prompt 位于 `docs/提示词设计/`
- 三种分析模式：translate（翻译）/ summary（摘要）/ review（综述）

### 应用核心流程
```
用户登录（SQLite 用户管理）
    ↓
侧边栏输入参数（keyword, max_results, analysis_type）
    ↓
search_pubmed() -> esearch API -> PMID 列表 -> efetch API -> XML 解析 -> 文献列表
    ↓
analyze_with_deepseek() -> 按 analysis_type 选择 System Prompt -> DeepSeek API -> Markdown 报告
    ↓
页面展示报告 + 下载按钮 + 记录用量
```

## 目录结构说明
```
streamlit/
├── CLAUDE.md                              # Agent 宪法（本文件）
├── README.md                              # 文件索引说明
├── .trae/rules/project_rules.md           # 项目规则
├── .streamlit/config.toml                 # Streamlit 配置
├── app.py                                 # 主应用（UI + 检索 + 分析逻辑）
├── db.py                                  # 数据库模块（用户管理 + 用量记录）
├── requirements.txt                       # Python 依赖
├── docs/                                  # 项目文档
│   ├── 需求分析报告/                       # 需求分析报告
│   ├── 可行性评估报告/                     # 技术可行性评估
│   ├── 架构设计/                           # 架构设计 + 模块配置手册
│   ├── 提示词设计/                         # DeepSeek LLM 提示词（翻译/摘要/综述）
│   └── 使用说明书/                         # 用户使用说明书
└── scripts/                               # 工具脚本（md2pdf 等）
```

## 全局代码规范
1. **Python 代码**：函数必须有类型注解，异常处理使用 try/except，避免裸 except
2. **提示词**：System Prompt 必须包含输出格式示例，使用 `{变量名}` 模板语法（Python str.format）
3. **PubMed API**：使用 E-utilities API（esearch + efetch），注意频率限制（3 次/秒无 API Key，10 次/秒有 API Key）
4. **DeepSeek API**：使用 OpenAI 兼容 SDK 调用，base_url 为 `https://api.deepseek.com`
5. **数据库**：SQLite 内置 sqlite3，无需额外依赖，首次运行自动建表

## 三阶工作流（必须遵守）
任何修改前必须按以下顺序执行：

### 1. Explore（探索阶段）- 只读不修改
- 先阅读 app.py 和 db.py，理解当前结构和逻辑
- 输出分析结论：目标模块的结构、依赖、核心逻辑

### 2. Plan（规划阶段）- 输出方案
- 明确列出要改动的文件、影响范围、风险点
- 确认方案后再动手

### 3. Implement（执行阶段）- 分步落地
- 按规划分步修改
- 每完成一步先自查，再继续下一步
- 完成后执行验证（streamlit run app.py）

## Stop Rules（禁止操作）
- 不随意改动 app.py 中的核心函数签名（search_pubmed, analyze_with_deepseek）
- 不删除 db.py 中的用户管理逻辑
- 不修改需求分析报告和可行性评估报告内容（只读参考）
- PubMed 检索结果必须标注来源（PMID、DOI），禁止编造文献

## 核心避坑点
1. **PubMed API 频率限制**：无 API Key 限 3 次/秒，有 API Key 限 10 次/秒，批量检索需加延迟
2. **摘要 vs 全文**：PubMed API 免费提供摘要，全文需 publisher 权限，应用默认处理摘要
3. **DeepSeek 输出格式**：必须用结构化输出（Markdown），避免自由散文式输出
4. **XML 解析**：AbstractText 可能多段且带 Label 属性，需拼接处理；ArticleTitle 可能含子标签，用 itertext 拼接
5. **DeepSeek API Key**：通过环境变量 DEEPSEEK_API_KEY 或侧边栏输入，不可硬编码
6. **SQLite 并发**：Streamlit 多用户并发时 SQLite 写入需注意，当前每次操作独立连接
7. **中英翻译质量**：医学术语需保留原文对照，不可随意意译

## 提示词设计规范
DeepSeek 的 System Prompt 使用结构化格式（与 docs/提示词设计/ 一致）：
- **角色定义**：明确 AI 扮演的专业角色（医学翻译/文献分析专家）
- **任务说明**：具体要完成的任务描述
- **输入格式**：明确输入数据的结构和字段
- **输出格式**：给出明确的输出模板/示例
- **限制条件**：禁止违反的规则（如不得编造文献、不得遗漏关键数据）

医学文献场景特别注意：
- 翻译时保留关键医学术语原文对照（如 "myocardial infarction（心肌梗死）"）
- 文献摘要必须标注 PMID 和来源
- 综述整理必须基于实际检索到的文献，禁止编造
- 输出语言默认中文（用户可指定英文）

## 目标定义规范（Loop Engineering）
每次修改前，明确回答以下问题：
1. **完成标准**：什么算"做完"？（可机器验证的条件）
2. **边界条件**：哪些不能碰？
3. **失败降级**：如果做不成怎么办？
4. **验收方式**：怎么验证做对了？
