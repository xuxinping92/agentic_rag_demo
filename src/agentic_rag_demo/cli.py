from __future__ import annotations

import argparse

from .runner import run_query


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic RAG demo")
    parser.add_argument("query", nargs="*", help="your question")
    args = parser.parse_args()
    query = " ".join(args.query) if args.query else "什么是 Agentic RAG？"
    result = run_query(query)
    print(result)


if __name__ == "__main__":
    main()
