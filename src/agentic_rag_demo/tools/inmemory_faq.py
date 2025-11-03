from __future__ import annotations

from .base import Tool

_FAQ = [
    "Agentic RAG 是在传统 RAG 上加入 agent 思考/规划/工具调用的一种架构，用来动态选择检索策略。",
    "传统 RAG 一般是 query -> embed -> vectorDB -> LLM，而 Agentic RAG 可能是 refine -> plan -> retrieve(xN) -> verify -> answer。",
    "Agentic RAG 的关键是把检索当成一个推理过程，而不是一次性召回。",
]


class InMemoryFAQTool(Tool):
    name = "inmemory_faq"
    description = "内置的简单 FAQ 检索，避免没装向量库也可以跑。"

    def run(self, query: str) -> str:
        # naive match
        for item in _FAQ:
            if "RAG" in query or "rag" in query.lower():
                return item
        # default
        return _FAQ[0]
