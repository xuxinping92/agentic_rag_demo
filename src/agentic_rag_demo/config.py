from __future__ import annotations
import os
from pydantic import BaseModel, Field

class LLMConfig(BaseModel):
    provider: str = Field(default="auto")  # auto|openai|ollama|dummy
    model: str = Field(default="gpt-4o-mini")
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
    # auto detect qdrant
    if os.environ.get("QDRANT_URL"):
        cfg.rag.use_qdrant = True
        cfg.rag.qdrant_url = os.environ["QDRANT_URL"]
    return cfg