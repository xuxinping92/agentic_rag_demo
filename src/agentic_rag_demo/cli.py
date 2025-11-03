from __future__ import annotations
import argparse
import os
from .runner import run_query


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic RAG demo")
    parser.add_argument("query", nargs="*", help="your question")
    parser.add_argument(
        "--verbose", action="store_true", help="print intermediate steps"
    )
    args = parser.parse_args()

    if args.verbose:
        os.environ["RAG_VERBOSE"] = "1"

    query = " ".join(args.query) if args.query else "什么是 Agentic RAG？"
    result = run_query(query)
    print(result)


if __name__ == "__main__":
    main()
