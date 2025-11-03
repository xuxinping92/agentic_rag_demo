from __future__ import annotations
import os
import json
import requests
from .config import LLMConfig


class LLMClient:
    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self.ollama_base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self.provider = self._detect_provider(cfg)

    def _detect_provider(self, cfg: LLMConfig) -> str:
        if cfg.provider != "auto":
            return cfg.provider
        if os.environ.get("OPENAI_API_KEY"):
            return "openai"
        # try ollama
        try:
            requests.get(f"{self.ollama_base}/api/tags", timeout=1)
            return "ollama"
        except Exception:
            return "dummy"

    def chat(self, messages: list[dict]) -> str:
        if self.provider == "openai":
            return self._chat_openai(messages)
        elif self.provider == "ollama":
            try:
                return self._chat_ollama(messages)
            except Exception as e:
                return f"【Dummy LLM(ollama回退)】{e.__class__.__name__}: {e}"
        else:
            # dummy
            parts = [m["content"] for m in messages if m["role"] == "user"]
            return "【Dummy LLM】我收到了你的问题: " + " / ".join(parts)

    # -------- OpenAI ----------
    def _chat_openai(self, messages: list[dict]) -> str:
        api_key = os.environ["OPENAI_API_KEY"]
        url = os.environ.get(
            "OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions"
        )
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
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    # -------- Ollama ----------
    def _pick_ollama_model(self) -> str:
        # 优先使用环境变量指定
        env_model = os.environ.get("OLLAMA_MODEL")
        if env_model:
            return env_model
        # 否则用配置里的，否则根据已安装列表挑一个
        preferred = self.cfg.model or "deepseek-r1:8b"
        try:
            tags = requests.get(f"{self.ollama_base}/api/tags", timeout=3).json()
            models = [m.get("name") for m in tags.get("models", []) if m.get("name")]
        except Exception:
            models = []
        if preferred in (models or []):
            return preferred
        # 常见可用模型优先
        for cand in ["deepseek-r1:8b", "qwen2.5:7b-instruct", "llama3.1:8b-instruct"]:
            if cand in models:
                return cand
        # 最后退而求其次
        return models[0] if models else preferred

    def _chat_ollama(self, messages: list[dict]) -> str:
        model = self._pick_ollama_model()

        # 1) 先尝试 /api/chat
        chat_url = f"{self.ollama_base}/api/chat"
        payload = {"model": model, "messages": messages, "stream": False}
        resp = requests.post(chat_url, json=payload, timeout=120)
        if resp.status_code in (404, 405):
            # 2) 回退到 /api/generate（把messages拼成一个prompt）
            prompt = self._messages_to_prompt(messages)
            gen_url = f"{self.ollama_base}/api/generate"
            gen_payload = {"model": model, "prompt": prompt, "stream": False}
            gen_resp = requests.post(gen_url, json=gen_payload, timeout=120)
            gen_resp.raise_for_status()
            data = gen_resp.json()
            # /api/generate 返回字段叫 "response"
            return (data.get("response") or "").strip()

        resp.raise_for_status()
        data = resp.json()
        # /api/chat 返回字段为 message.content
        if isinstance(data, dict) and "message" in data:
            return data["message"]["content"].strip()
        return str(data)

    @staticmethod
    def _messages_to_prompt(messages: list[dict]) -> str:
        """把 Chat messages 转成一个纯文本 prompt 给 /api/generate 用。"""
        lines = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                lines.append(f"[系统指令]\n{content}\n")
            elif role == "assistant":
                lines.append(f"助手：{content}\n")
            else:
                lines.append(f"用户：{content}\n")
        lines.append("请基于以上对话继续回复。")
        return "\n".join(lines)
