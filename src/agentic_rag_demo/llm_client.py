from __future__ import annotations
import os
import json
import requests
from .config import LLMConfig

class LLMClient:
    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self.provider = self._detect_provider(cfg)

    def _detect_provider(self, cfg: LLMConfig) -> str:
        if cfg.provider != "auto":
            return cfg.provider
        if os.environ.get("OPENAI_API_KEY"):
            return "openai"
        # try ollama
        try:
            requests.get("http://localhost:11434/api/tags", timeout=0.5)
            return "ollama"
        except Exception:
            return "dummy"

    def chat(self, messages: list[dict]) -> str:
        if self.provider == "openai":
            return self._chat_openai(messages)
        elif self.provider == "ollama":
            return self._chat_ollama(messages)
        else:
            # dummy
            parts = [m["content"] for m in messages if m["role"] == "user"]
            return "【Dummy LLM】我收到了你的问题: " + " / ".join(parts)

    def _chat_openai(self, messages: list[dict]) -> str:
        api_key = os.environ["OPENAI_API_KEY"]
        url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions")
        model = self.cfg.model or "gpt-4o-mini"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": self.cfg.temperature,
        }
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    def _chat_ollama(self, messages: list[dict]) -> str:
        # pick first local model
        model = self.cfg.model or "deepseek-r1:8b"
        url = "http://localhost:11434/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"].strip()