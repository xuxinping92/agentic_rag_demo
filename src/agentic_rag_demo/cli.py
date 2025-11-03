from __future__ import annotations

import argparse

from .runner import run_query


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic RAG demo")
    parser.add_argument("query", nargs="*", help="your question")
    parser.add_argument(
        "--recall",
        type=int,
        default=None,
        help="检索(召回)轮数上限，默认读取配置 rag.recall_rounds",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="指定使用的 LLM 模型，例如 deepseek-r1:8b、qwen2:7b、gpt-4o 等",
    )

    args = parser.parse_args()
    query = " ".join(args.query) if args.query else "什么是 Agentic RAG？"

    result = run_query(query, recall_rounds=args.recall, model=args.model)
    print(result)


if __name__ == "__main__":
    main()
