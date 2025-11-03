# agentic_rag_demo

这是一个最小可跑的 Agentic RAG 演示包，吸收了《LLM之RAG实战（五十八）》《（五十九）》里的思路，方便你在本机（Win + Python）直接测试。

## 特点

- 内置一个简单的“FAQ 向量库”（内存），开箱即用；
- 如果本机有 Qdrant / sentence-transformers，可自动切换到真向量库；
- 提供一个简单的 agent 流程：**refine → retrieve → generate → reflect**；
- LLM 可选：
  - 环境变量里有 `OPENAI_API_KEY` → 走 OpenAI
  - 否则尝试调用本机 Ollama (`http://localhost:11434`) 的 deepseek / qwen
  - 再不行就用一个假的 LLM 回答，便于你看流程

## 安装

```bash
cd agentic_rag_demo
pip install -e .
```

## 运行

```bash
agentic-rag-demo "什么是 Agentic RAG？"
```

也可以运行示例：

```bash
python examples/run_demo.py
```