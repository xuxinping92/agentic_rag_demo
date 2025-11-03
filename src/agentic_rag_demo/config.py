# src/agentic_rag_demo/config.py
from __future__ import annotations

import os

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    provider: str = Field(default="ollama")  # auto|openai|ollama|dummy
    model: str = Field(default="qwen2.5:1.5b-instruct")
    temperature: float = 0.2


class RAGConfig(BaseModel):
    use_qdrant: bool = False
    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "agentic-rag-demo"
    top_k: int = 3
    recall_rounds: int = 1  # 召回(检索)轮数上限，用于控制运行时长


class AppConfig(BaseModel):
    llm: LLMConfig = LLMConfig()
    rag: RAGConfig = RAGConfig()


def load_config() -> AppConfig:
    cfg = AppConfig()

    # --- 新增：允许用环境变量覆盖 LLM 配置 ---
    if os.environ.get("LLM_PROVIDER"):
        cfg.llm.provider = os.environ["LLM_PROVIDER"]  # e.g. "ollama"
    if os.environ.get("LLM_MODEL"):
        cfg.llm.model = os.environ["LLM_MODEL"]  # e.g. "qwen2.5:1.5b-instruct"
    if os.environ.get("LLM_TEMPERATURE"):
        try:
            cfg.llm.temperature = float(os.environ["LLM_TEMPERATURE"])
        except ValueError:
            pass

    # 保持你原来的 Qdrant 自动检测
    if os.environ.get("QDRANT_URL"):
        cfg.rag.use_qdrant = True
        cfg.rag.qdrant_url = os.environ["QDRANT_URL"]

    return cfg
