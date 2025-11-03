from __future__ import annotations

from .base import Tool

_FAQ = [
    "Agentic RAG 是在传统 RAG 上加入 agent 思考/规划/工具调用的一种架构，用来动态选择检索策略。",
    "传统 RAG 一般是 query -> embed -> vectorDB -> LLM，而 Agentic RAG 可能是 refine -> plan -> retrieve(xN) -> verify -> answer。",
    "Agentic RAG 的关键是把检索当成一个推理过程，而不是一次性召回。",
    # "北京十日游：D1 故宫-景山；D2 颐和园-圆明园；D3 八达岭长城；D4 天坛-前门大街；D5 国家博物馆-王府井；D6 北海-什刹海-南锣鼓巷；D7 恭王府-孔庙国子监；D8 中轴线打卡；D9 798 艺术区；D10 清华北大外观与周边。",
    # "北京出行提示：热门景点需预约；避开早晚高峰地铁；长城注意防晒与穿舒适鞋；博物馆周一闭馆要留意；推荐电子支付与离线地图。",
]


class InMemoryFAQTool(Tool):
    name = "inmemory_faq"
    description = "内置的简单 FAQ 检索，避免没装向量库也可以跑。"

    def run(self, query: str) -> str:
        # naive match
        for item in _FAQ:
            if "RAG" in query or "rag" in query.lower():
                return item
        # default
        return _FAQ[0]
