from __future__ import annotations
from .config import load_config
from .llm_client import LLMClient
from .tools.inmemory_faq import InMemoryFAQTool
from .tools.qdrant_faq import QdrantFAQTool

def run_query(query: str) -> str:
    cfg = load_config()
    llm = LLMClient(cfg.llm)

    # 1) refine query
    refined = llm.chat([
        {"role": "system", "content": "你是一个查询优化器，负责把用户的口语问题改写成适合检索的短句。"},
        {"role": "user", "content": query},
    ])

    # 2) choose tool
    if cfg.rag.use_qdrant:
        tool = QdrantFAQTool(cfg.rag.qdrant_url, cfg.rag.collection_name)
        ctx = tool.run(refined)
        if "回退" in ctx or "失败" in ctx:
            ctx = InMemoryFAQTool().run(refined)
    else:
        tool = InMemoryFAQTool()
        ctx = tool.run(refined)

    # 3) generate final answer
    answer = llm.chat([
        {"role": "system", "content": "你是一个知识助手，会基于给定的上下文回答问题，不能胡编。"},
        {"role": "user", "content": f"用户问题: {query}\n\n已检索到的上下文: {ctx}\n\n请用中文给出最终回答。"},
    ])

    # 4) reflect
    reflection = llm.chat([
        {"role": "system", "content": "你是一个自我评估器，检查回答是否跑题，如果有问题给出改进版，否则返回原文。"},
        {"role": "user", "content": answer},
    ])

    return reflection