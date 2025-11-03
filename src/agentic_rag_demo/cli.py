from __future__ import annotations

import argparse
import os

from .runner import run_query


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic RAG demo")
    parser.add_argument("query", nargs="*", help="your question")
    parser.add_argument("--provider", choices=["auto", "openai", "ollama", "dummy"])
    parser.add_argument("--model")
    args = parser.parse_args()

    # 通过环境变量把参数传给 load_config()
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
    if args.model:
        os.environ["LLM_MODEL"] = args.model

    query = " ".join(args.query) if args.query else "什么是 Agentic RAG？"
    result = run_query(query)
    print(result)


if __name__ == "__main__":
    main()
