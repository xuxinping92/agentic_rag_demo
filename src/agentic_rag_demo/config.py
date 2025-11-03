from __future__ import annotations
import os
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    # provider: auto | openai | ollama | dummy
    provider: str = Field(default="auto")
    model: str = Field(
        default="gpt-4o-mini"
    )  # 可改成你的本地模型，如 "qwen2.5:7b-instruct"
    temperature: float = 0.2


class RAGConfig(BaseModel):
    use_qdrant: bool = False
    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "agentic-rag-demo"
    top_k: int = 3


class AppConfig(BaseModel):
    llm: LLMConfig = LLMConfig()
    rag: RAGConfig = RAGConfig()


def load_config() -> AppConfig:
    cfg = AppConfig()
    # 自动探测 Qdrant
    if os.environ.get("QDRANT_URL"):
        cfg.rag.use_qdrant = True
        cfg.rag.qdrant_url = os.environ["QDRANT_URL"]
    return cfg
