from __future__ import annotations

from typing import Optional

from .base import Tool

try:
    from qdrant_client import QdrantClient, models
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    QdrantClient = None
    SentenceTransformer = None


class QdrantFAQTool(Tool):
    name = "qdrant_faq"
    description = "使用 Qdrant + sentence-transformers 的真实检索（如果本机装了的话）"

    def __init__(self, url: str, collection: str = "agentic-rag-demo") -> None:
        self.available = QdrantClient is not None and SentenceTransformer is not None
        self.url = url
        self.collection = collection
        if self.available:
            self.client = QdrantClient(url=url)
            self.embedder = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2"
            )
        else:
            self.client = None
            self.embedder = None

    def run(self, query: str) -> str:
        if not self.available:
            return "Qdrant 未安装或未启动，回退到内存 FAQ。"
        vec = self.embedder.encode(query).tolist()
        try:
            hits = self.client.search(
                collection_name=self.collection,
                query_vector=vec,
                limit=3,
            )
            if not hits:
                return "Qdrant 里没搜到内容。"
            parts = [h.payload.get("text", "") for h in hits]
            return "\n".join(parts)
        except Exception as e:
            return f"Qdrant 检索失败: {e}"
