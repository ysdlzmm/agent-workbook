import os

from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

"""搜索工具函数"""


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

        # 如果没有直接总结，拼接多条摘要
        snippets = []
        for i, res in enumerate(results.get("results", [])):
            snippets.append(
                f"[{i + 1}] {res.get('title', '')}\n{res.get('content', '')}"
            )
        raw_text = "\n\n".join(snippets)
        return raw_text

    except Exception as e:
        return f"搜索时发生错误：{e}"


"""ToolExecutor 统一的管理器来注册和调度这些工具"""
from typing import Dict, Any


class ToolExecutor:
    """
    一个工具执行器，负责管理和执行工具
    """

    def __init__(self):
        self.tools: Dict[str, Any] = {}

    def registerTool(self, name: str, description: str, func: callable):
        """
        向工具箱中注册一个工具
        """
        if name in self.tools:
            print(f"⚠️警告：工具 '{name}' 已经注册，将覆盖原有工具。")
        self.tools[name] = {"description": description, "func": func}
        print(f"✅工具 '{name}' 注册成功。")

    def getTool(self, name: str) -> callable:
        """
        根据名称获取一个工具的执行函数
        """
        return self.tools.get(name, {}).get("func")

    def getAvailableTools(self) -> str:
        """
        获取所有可用工具的格式化描述字符串
        """

        return "\n".join(
            [f"【{name}】: {info['description']}" for name, info in self.tools.items()]
        )


# --- 工具初始化与使用示例 ---
if __name__ == "__main__":
    # 1. 初始化工具执行器
    toolExecutor = ToolExecutor()

    # 2. 注册搜索工具
    search_description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    toolExecutor.registerTool("Search", search_description, search)

    # 3. 打印可用的工具
    print("\n--- 可用的工具 ---")
    print(toolExecutor.getAvailableTools())

    # 4. 智能体的Action调用，这次我们问一个实时性的问题
    print("\n--- 执行Action: Search['英伟达最新的GPU型号是什么'] ---")
    tool_name = "Search"
    tool_input = "英伟达最新的GPU型号是什么"

    tool_function = toolExecutor.getTool(tool_name)
    if tool_function:
        observation = tool_function(tool_input)
        print("--- 观察 （Observation） ---")
        print(observation)
    else:
        print("--- 🙅错误：未找到名为 '{tool_name}'的工具 ---")
