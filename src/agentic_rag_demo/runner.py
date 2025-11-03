from __future__ import annotations

import json
import os

from .config import load_config
from .llm_client import LLMClient
from .tools.inmemory_faq import InMemoryFAQTool
from .tools.qdrant_faq import QdrantFAQTool

VERBOSE = os.environ.get("RAG_VERBOSE", "0") == "1"


def _vprint(*args):
    if VERBOSE:
        print(*args)


def run_query(query: str) -> str:
    cfg = load_config()
    llm = LLMClient(cfg.llm)

    # 1) refine query
    refined = llm.chat(
        [
            {
                "role": "system",
                "content": "你是查询优化器，把用户问题改写成更利于检索的简洁短句。只输出改写后的短句。",
            },
            {"role": "user", "content": query},
        ]
    )
    _vprint("== Refined Query ==\n", refined, "\n")

    # 2) choose tool & retrieve
    if cfg.rag.use_qdrant:
        tool = QdrantFAQTool(cfg.rag.qdrant_url, cfg.rag.collection_name)
        ctx = tool.run(refined)
        if "回退" in ctx or "失败" in ctx:
            ctx = InMemoryFAQTool().run(refined)
    else:
        tool = InMemoryFAQTool()
        ctx = tool.run(refined)
    _vprint("== Retrieved Context ==\n", ctx, "\n")

    # 3) draft answer
    answer = llm.chat(
        [
            {
                "role": "system",
                "content": "你是一个知识助手，会基于给定的上下文回答问题，不能胡编。",
            },
            {
                "role": "user",
                "content": f"用户问题: {query}\n\n已检索到的上下文: {ctx}\n\n请用中文给出清晰的最终回答。",
            },
        ]
    )
    _vprint("== Draft Answer ==\n", answer, "\n")

    # 4) reflect with STRICT JSON
    reflect_prompt = [
        {
            "role": "system",
            "content": (
                "你是答案质检器。请只输出严格的 JSON（不要任何解释文字）。"
                'JSON 格式为：{"verdict":"keep|rewrite","final_answer":"..."}。'
                '规则：若答案偏题/事实错误/不清楚，则 verdict= "rewrite" 并在 final_answer 给出改进后的完整答案；'
                '否则 verdict= "keep"，且 final_answer 必须原样返回草稿答案全文。'
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "question": query,
                    "context": ctx,
                    "draft_answer": answer,
                },
                ensure_ascii=False,
            ),
        },
    ]
    reflection_raw = llm.chat(reflect_prompt)
    _vprint("== Reflection Raw ==\n", reflection_raw, "\n")

    final_answer = answer  # fallback
    try:
        data = json.loads(reflection_raw)
        if isinstance(data, dict) and "final_answer" in data:
            final_answer = data["final_answer"] or answer
    except Exception:
        # 如果模型没按 JSON 返回，就直接用草稿答案
        pass

    return final_answer.strip()
