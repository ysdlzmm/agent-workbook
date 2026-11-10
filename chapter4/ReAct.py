"""Tavily 网页搜索工具（chapter4 示例）。"""

import os
import json

from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()


def search(query: str) -> str:
    """
    一个基于 Tavily 的实战网页搜索引擎工具。
    它会智能地解析搜索结果，优先返回直接答案或知识图谱信息。
    """
    print(f"🔍 正在执行 [Tavily] 网页搜索：{query}")
    try:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return "TAVILY_API_KEY 未在 .env 文件中配置。"

        client = TavilyClient(api_key=api_key)
        results = client.search(
            query=query,
            max_results=3,
            topic="general",
            include_answer=True,
            language="zh-cn",  # 优先中文
            filter_by_language=True,  # 严格过滤，丢掉非中文页面
        )

        # 优先拿 Tavily 自带的总结 answer
        if results.get("answer"):
            return results["answer"]

        # 如果没有直接总结，拼接多条摘要
        snippets = []
        for i, res in enumerate(results.get("results", [])):
            snippets.append(
                f"[来源{i + 1}] {res.get('title', '')}\n{res.get('content', '')}"
            )
        raw_text = "\n\n".join(snippets)
        return raw_text

    except Exception as e:
        return f"搜索时发生错误：{e}"


if __name__ == "__main__":
    print(search("华为最新手机"))
