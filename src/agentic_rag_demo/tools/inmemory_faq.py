from __future__ import annotations

import re
from collections import Counter
from typing import List, Tuple

from .base import Tool

_FAQ = [
    # Agentic RAG 基础与原理
    "Agentic RAG 是在传统 RAG 上加入 agent 思考、规划、工具调用的一种架构，用来动态选择检索策略。",
    "传统 RAG 通常是固定流程：query -> embed -> vectorDB -> LLM，而 Agentic RAG 则可能是 refine -> plan -> retrieve(xN) -> verify -> answer。",
    "Agentic RAG 的关键是将信息检索视为一个推理过程，而不是一次性召回。",
    "传统 RAG 更像是一个图书馆员，只根据关键字检索内容；Agentic RAG 则像一个智能助手，会先思考、再规划、再验证结果。",
    "Agentic RAG 的核心理念是：反思（Reflection）、规划（Planning）、工具使用（Tool Use）和多代理协作（Multi-Agent Collaboration）。",
    "在复杂查询中，Agentic RAG 可以动态决定使用向量数据库、Web搜索、API工具或多轮推理等方式，从而得到更高质量的答案。",
    # 传统 RAG 的问题
    "传统 RAG 存在‘分块窘境’：文本分块太小会丢失上下文，太大会降低检索效率。",
    "嵌入模型的不精确性可能导致召回错误结果，例如把‘苹果股崩盘’误解为‘苹果水果掉落’。",
    "传统向量数据库（如 Pinecone、Weaviate）虽强大，但在索引更新与成本上复杂且昂贵。",
    "LLM 的上下文窗口有限（如 GPT-4 为128K tokens），过多或不相关的块会导致幻觉或忽略关键信息。",
    "传统 RAG 对简单问答有效，但在多步骤或推理型问题中效果不佳。",
    # Agentic RAG 技术机制
    "Agentic RAG 中的代理组件会先‘导航’文档，优化查询并决定使用哪些工具或检索方式。",
    "代理可以通过调用外部API、网络搜索或多个向量数据库来整合答案。",
    "LangChain 或 LangGraph 等框架可用于构建 Agentic RAG 工作流，实现节点式的 refine→retrieve→generate 流程。",
    "Agentic RAG 可以结合反思机制：对初步答案进行自我审查（Reflection），若不满意则重新检索。",
    # 优缺点与适用场景
    "Agentic RAG 的优点包括：动态检索、自我更正、多步骤推理和工具集成。",
    "其缺点包括：响应延迟更高、成本上升、系统复杂度高、多代理协调困难。",
    "Agentic RAG 适合处理复杂或多步骤的查询、实时或动态数据场景、高精度任务。",
    "若数据静态且问题简单，则传统 RAG 更高效。",
    # MCP 架构与实现
    "MCP（模型上下文协议，Model Context Protocol）是一种开放标准，让LLM能安全访问外部数据、工具与服务。",
    "MCP 相当于 AI 的 USB-C 接口：统一标准、双向通信、安全可控。",
    "MCP 体系包括 MCP 服务器（提供工具与数据）和 MCP 客户端（调用与交互）。",
    "在基于 MCP 的 Agentic RAG 架构中，MCP 服务器扮演智能代理的大脑，能够判断何时使用矢量数据库、网络搜索或两者结合。",
    "FireCrawl Web 搜索工具让代理可以实时获取互联网数据；而 Qdrant 向量数据库用于私有知识检索。",
    "通过 @mcp_server.tool() 装饰器，可以向 MCP 注册工具函数，代理会根据文档字符串自动选择合适工具。",
    "FAQEngine 类基于 HuggingFace Embedding 与 QdrantClient 实现嵌入与检索，用于 RAG 管道核心逻辑。",
    "Agentic RAG 管道包括数据索引（setup_collection）、嵌入生成、向量化存储、语义检索与上下文增强。",
    "使用 MCP 时，代理会自动在 Python FAQ 数据库与 Web 搜索之间切换，以回答用户不同类型问题。",
    "Qdrant 数据库通过 Docker 可快速部署，端口6333用于API，6334用于Web UI。",
    # 应用与发展方向
    "Agentic RAG 并非取代向量数据库，而是将其变成智能生态系统的一部分。",
    "未来 Agentic RAG 将支持语音代理、浏览器代理等多模态应用场景。",
    "LangGraph 框架简化了 Agentic RAG 的可视化构建与状态流转。",
    "通过 MCP 可轻松扩展更多工具，如计算器、数据库查询、外部文档加载等，构建模块化智能体系统。",
    # "北京十日游：D1 故宫-景山；D2 颐和园-圆明园；D3 八达岭长城；D4 天坛-前门大街；D5 国家博物馆-王府井；D6 北海-什刹海-南锣鼓巷；D7 恭王府-孔庙国子监；D8 中轴线打卡；D9 798 艺术区；D10 清华北大外观与周边。",
    # "北京出行提示：热门景点需预约；避开早晚高峰地铁；长城注意防晒与穿舒适鞋；博物馆周一闭馆要留意；推荐电子支付与离线地图。",
]


def _normalize(s: str) -> str:
    # 简单归一化：小写、去标点、压缩空白
    s = s.lower()
    s = re.sub(r"[^\w\u4e00-\u9fff]+", " ", s)  # 保留英文/数字/中文
    s = re.sub(r"\s+", " ", s).strip()
    return s


# 领域关键词加权（可按需增删）
_KEY_WEIGHTS = {
    "agentic": 2.0,
    "rag": 2.0,
    "mcp": 2.0,
    "langgraph": 1.6,
    "langchain": 1.2,
    "向量": 1.2,
    "嵌入": 1.2,
    "qdrant": 1.6,
    "pinecone": 1.0,
    "weaviate": 1.0,
    "工具": 1.0,
    "反思": 1.0,
    "规划": 1.0,
    "多代理": 1.0,
    "检索": 1.2,
    "幻觉": 1.0,
}


def _score_item(item: str, q_norm: str, q_tokens: Counter) -> float:
    txt = _normalize(item)
    score = 0.0
    # 1) 词重叠
    item_tokens = Counter(txt.split())
    overlap = sum(min(q_tokens[t], item_tokens[t]) for t in q_tokens)
    score += overlap

    # 2) 关键词加权（中英双活）
    for k, w in _KEY_WEIGHTS.items():
        if k in txt:
            score += w
        if k in q_norm:
            score += 0.2 * w  # 查询包含关键词也加一点点，平滑

    # 3) 子串强命中奖励
    if q_norm and q_norm in txt:
        score += 2.5

    return score


class InMemoryFAQTool(Tool):
    name = "inmemory_faq"
    description = "内置的轻量 FAQ 检索（不依赖向量库），适合本地快速演示。"

    def run(self, query: str, top_k: int = 3) -> str:
        if not _FAQ:
            return "FAQ 为空。"
        q = query or ""
        q_norm = _normalize(q)
        q_tokens = Counter(q_norm.split()) if q_norm else Counter()

        # 对每条 FAQ 评分
        scored: List[Tuple[float, str]] = [
            (_score_item(item, q_norm, q_tokens), item) for item in _FAQ
        ]
        scored.sort(key=lambda x: x[0], reverse=True)

        # 选前 top_k；若最高分很低（比如完全不相关），仍然返回第一条作为兜底
        best_k = [it for _, it in scored[: max(1, top_k)]]
        # 去重保持顺序
        seen, deduped = set(), []
        for it in best_k:
            if it not in seen:
                seen.add(it)
                deduped.append(it)

        # 友好格式化输出
        if len(deduped) == 1:
            return deduped[0]
        return "；\n".join(f"- {s}" for s in deduped)

    # 可选：允许运行期动态扩充 FAQ
    def add_faq(self, items: List[str]) -> int:
        n0 = len(_FAQ)
        for it in items:
            if isinstance(it, str) and it.strip():
                _FAQ.append(it.strip())
        return len(_FAQ) - n0
