from __future__ import annotations

import json
import os

from .config import load_config
from .llm_client import LLMClient
from .tools.inmemory_faq import InMemoryFAQTool
from .tools.qdrant_faq import QdrantFAQTool


def _is_verbose() -> bool:
    return os.environ.get("RAG_VERBOSE", "0") == "1"


def _vprint(*args):
    if _is_verbose():
        print(*args)


def run_query(
    query: str, *, recall_rounds: int | None = None, model: str | None = None
) -> str:
    """
    Agentic RAG 四步流程：
        1) refine 查询
        2) retrieve 检索（支持多轮召回 recall_rounds）
        3) draft 初稿生成
        4) reflect 答案质检

    参数:
        query : 用户输入的问题
        recall_rounds : 检索轮数上限（默认从配置读取）
        model : 临时指定的模型名称（可覆盖配置）
    """
    cfg = load_config()
    if model:
        cfg.llm.model = model
    llm = LLMClient(cfg.llm)

    # ========== 1) refine query ==========
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

    # ========== 2) choose tool & multi-round retrieve ==========
    if recall_rounds is None:
        recall_rounds = getattr(cfg.rag, "recall_rounds", 1)
    if recall_rounds < 1:
        recall_rounds = 1

    if cfg.rag.use_qdrant:
        base_tool = QdrantFAQTool(cfg.rag.qdrant_url, cfg.rag.collection_name)
    else:
        base_tool = InMemoryFAQTool()

    ctx_chunks = []
    for i in range(recall_rounds):
        _vprint(f"== Retrieval Round {i+1}/{recall_rounds} ==")
        ctx_i = base_tool.run(refined)
        # 如果检索失败或触发回退逻辑，则用内存FAQ兜底
        if "回退" in ctx_i or "失败" in ctx_i:
            ctx_i = InMemoryFAQTool().run(refined)
        _vprint(f"== Retrieved Context (Round {i+1}) ==\n", ctx_i, "\n")

        if ctx_i:
            ctx_chunks.append(ctx_i)
        # 可选早停逻辑
        if any(stop_kw in ctx_i for stop_kw in ["未找到", "检索失败", "空结果"]):
            _vprint(f"停止于第 {i+1} 轮：检索结果为空或无效。")
            break

    ctx = "\n---\n".join(ctx_chunks) if ctx_chunks else ""
    _vprint("== Combined Context ==\n", ctx, "\n")

    # ========== 3) draft answer ==========
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

    # ========== 4) reflect with STRICT JSON ==========
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
