import asyncio
import os
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from tavily import TavilyClient

# 无论从哪个目录启动脚本，都固定读取项目根目录的 .env
ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


class SearchState(TypedDict):
    messages: Annotated[list, add_messages]
    user_query: str  # 经过LLM理解后的用户需求总结
    search_query: str  # 优化后用于Tavily API的搜索查询
    search_result: str  # Tavily搜索返回的结果
    final_answer: str  # 最终生成的答案
    step: str  # 标记当前步骤


# 初始化模型
# 我们将使用这个 llm 实例来驱动所有节点的智能
llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL_ID"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
    temperature=0.7,
)
# 初始化Tavily客户端
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# 理解与查询
def understand_query_node(state: SearchState) -> dict:
    """步骤1: 理解用户查询并生成搜索关键词"""
    user_message = state["messages"][-1].content

    understand_prompt = f"""分析用户的查询： "{user_message}"
    请完成两个任务：
    1. 简洁总结用户想要了解什么
    2. 生成最适合搜索引擎的关键词（中英文均可，要精准）

    格式：
    理解：[用户需求总结]
    搜索词：[最佳搜索关键词]
    """

    response = llm.invoke([SystemMessage(content=understand_prompt)])
    response_text = response.content

    # 解析LLM的输出，提取搜索关键词
    search_query = user_message  # 默认使用原始查询
    if "搜索词：" in response_text:
        search_query = response_text.split("搜索词：")[1].strip()

    return {
        "user_query": user_message,
        "search_query": search_query,
        "step": "understand_query",
        "messages": [AIMessage(content=f"我将为您搜索：{search_query}")],
    }


# 搜索节点
def tavily_search_node(state: SearchState) -> dict:
    """步骤2: 使用Tavily API进行真实搜索"""
    search_query = state["search_query"]
    try:
        print(f"🔍 正在搜索：{search_query}")
        response = tavily_client.search(
            query=search_query, search_depth="basic", max_results=5, include_answer=True
        )
        # 把 Tavily 返回的结果列表拼成一段供 LLM 阅读的文本
        parts = []
        if response.get("answer"):
            parts.append(f"概览：{response['answer']}")
        for item in response.get("results", []):
            parts.append(
                f"- {item.get('title', '')}\n"
                f"  {item.get('url', '')}\n"
                f"  {item.get('content', '')}"
            )
        search_result = "\n".join(parts) if parts else "未找到相关结果"

        return {
            "search_result": search_result,
            "step": "searched",
            "messages": [AIMessage(content="✅ 搜索完成！正在整理答案...")],
        }
    except Exception as e:
        return {
            "search_result": f"搜索失败：{e}",
            "step": "search_failed",
            "messages": [AIMessage(content="❌ 搜索遇到问题...")],
        }


# 回答节点
def generate_answer_node(state: SearchState) -> dict:
    """步骤3: 基于搜索结果生成最终答案"""
    if state["step"] == "search_failed":
        # 如果搜索失败，执行回退策略，基于LLM自身知识回答
        fallback_prompt = f"搜索API暂时不可用，请基于您的知识回答用户的问题：\n用户问题：{state['user_query']}\n"
        response = llm.invoke([SystemMessage(content=fallback_prompt)])
    else:
        # 搜索成功，基于搜索结果生成答案
        answer_prompt = f"""基于以下搜索结果为用户提供完整，准确的答案：
        用户问题：{state['user_query']}
        搜索结果：\n{state['search_result']}
        请综合搜索结果，提供准确，有用的回答...
        """
        response = llm.invoke([SystemMessage(content=answer_prompt)])

    return {
        "final_answer": response.content,
        "step": "completed",
        "messages": [AIMessage(content=response.content)],
    }


# 构建图
def create_search_assistant():
    workflow = StateGraph(SearchState)

    # 添加节点
    workflow.add_node("understand", understand_query_node)
    workflow.add_node("search", tavily_search_node)
    workflow.add_node("answer", generate_answer_node)

    # 设置线性流程
    workflow.add_edge(START, "understand")
    workflow.add_edge("understand", "search")
    workflow.add_edge("search", "answer")
    workflow.add_edge("answer", END)

    # 编译图：checkpointer 按 thread_id 保存多轮对话状态
    checkpointer = InMemorySaver()
    app = workflow.compile(checkpointer=checkpointer)
    return app


async def main():
    """主函数：运行智能搜索助手"""

    if not os.getenv("TAVILY_API_KEY"):
        print("❌ 请设置环境变量 TAVILY_API_KEY")
        return

    app = create_search_assistant()

    print("🔍 智能搜索助手启动！")
    print("我会使用Tavily API为您搜索最新、最准确的信息")
    print("支持各种问题：新闻、技术、知识问答等")
    print("(输入 'quit' 退出)\n")

    session_count = 0

    while True:
        user_input = input("🤔 您想了解什么: ").strip()

        if user_input.lower() in ["quit", "exit", "q", "退出"]:
            print("👋 感谢使用智能搜索助手，再见！")
            break

        if not user_input:
            continue

        session_count += 1
        config = {"configurable": {"thread_id": f"search-session-{session_count}"}}

        # 初始状态
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "user_query": "",
            "search_query": "",
            "search_result": "",
            "final_answer": "",
            "step": "start",
        }

        try:
            print("\n" + "=" * 60)

            # 执行工作流
            async for output in app.astream(initial_state, config=config):
                for node_name, node_output in output.items():
                    if "messages" in node_output and node_output["messages"]:
                        latest_message = node_output["messages"][-1]
                        if isinstance(latest_message, AIMessage):
                            if node_name == "understand":
                                print(f"🧠 理解阶段: {latest_message.content}")
                            elif node_name == "search":
                                print(f"🔍 搜索阶段: {latest_message.content}")
                            elif node_name == "answer":
                                print(f"\n💡 最终回答:\n{latest_message.content}")

            print("\n" + "=" * 60 + "\n")

        except Exception as e:
            print(f"❌ 发生错误: {e}")
            print("请重新输入您的问题。\n")


if __name__ == "__main__":
    asyncio.run(main())
