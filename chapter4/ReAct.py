import re

from tools import ToolExecutor, search
from llm_client import HelloAgentsLLM

# ReAct 提示词模版
# 注意：.format() 会把单个花括号当占位符，正文里不要出现裸露的 { }
REACT_PROMPT_TEMPLATE = """
请注意，你是一个有能力调用外部工具的智能助手。

可用工具如下:
{tools}

请严格按照以下格式回应，每次只输出两行，不要输出任何额外内容:

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你的行动，必须是下面两种写法之一

Action 的合法写法（直接照抄格式，不要加反引号、花括号、引号或任何解释）:
- Search[查询关键词]   调用搜索工具，例如: Search[华为最新手机 主要卖点]
- Finish[最终答案]     已经收集到足够信息时，必须用它输出答案并结束

重要约束:
- 工具名必须原样使用上面列出的名称，不要用花括号包起来，也不要写成 Markdown 代码块。
- Action 后面只能有一行，不要换行补充内容。
- 每次必须同时输出 Thought 和 Action 两行。

现在，请开始解决以下问题:
Question: {question}
History: {history}
"""


class ReActAgent:
    def __init__(
        self,
        llm_client: HelloAgentsLLM,
        tool_executor: ToolExecutor,
        max_steps: int = 5,
    ):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []

    def run(self, question: str):
        """
        运行ReAct智能体回答一个问题。
        """
        self.history = []  # 每次运行时重置历史记录
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1
            print(f"--- 第 {current_step} 步")

            # 1. 格式化提示词
            tools_desc = self.tool_executor.getAvailableTools()
            history_str = "\n".join(self.history)
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=tools_desc, question=question, history=history_str
            )

            # 2. 调用LLM进行思考
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(messages=messages)

            if not response_text:
                print("❌错误：LLM未能返回有效响应")
                break

            # 3. 解析LLM的输出
            thought, action = self._parse_output(response_text)

            if thought:
                print(f"🤔 思考：{thought}")

            if not action:
                print("⚠️警告：未能解析出有效的Action，流程终止。")
                break

            # 4. 执行Action
            if action.startswith("Finish"):
                # 如果是Finish指令，提取最终答案并结束
                match = re.search(r"Finish\[(.*)\]", action, re.DOTALL)
                final_answer = match.group(1).strip() if match else action
                print(f"🎉最终答案：{final_answer}")
                return final_answer

            tool_name, tool_input = self._parse_action(action)
            if not tool_name or not tool_input:
                print(f"❌ 无效的Action: {action}")
                # 把失败反馈写进历史，否则下一轮提示词完全相同，模型会一直重复同一句
                self.history.append(f"Action: {action}")
                self.history.append(
                    "Observation: 无效的 Action。请严格输出 Search[查询关键词] "
                    "或 Finish[最终答案]，不要加花括号、反引号或多余文字。"
                )
                continue

            print(f"🎬行动：{tool_name}[{tool_input}]")

            tool_function = self.tool_executor.getTool(tool_name)
            if not tool_function:
                observation = f"❌ 错误：未找到工具 '{tool_name}'"
            else:
                observation = tool_function(tool_input)  # 调用真实工具

            print(f"👀观察：{observation}")

            # 将本轮的Action和Observation添加到历史记录中
            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")

        # 循环结束
        print("\n--- 已达到最大步数，流程终止。 ---")
        return None

    @staticmethod
    def _clean_action(action: str) -> str:
        """
        清洗模型输出里常见的 Markdown / 转义噪声。
        例如 `{Search}[关键词]`\\  ->  Search[关键词]
        """
        if not action:
            return ""
        # 只取第一行，丢掉模型补充的额外说明
        action = action.strip().splitlines()[0].strip()
        # 去掉 Markdown 反引号（可能在反斜杠内侧或外侧）、行尾转义符、包裹引号
        action = action.replace("`", "").strip().rstrip("\\").strip()
        action = action.strip("'\"").strip()
        # {Search[xxx]} -> Search[xxx]
        while action.startswith("{") and action.endswith("}"):
            action = action[1:-1].strip()
        # {Search}[xxx] -> Search[xxx]
        action = re.sub(r"^\{(\w+)\}", r"\1", action)
        return action.strip()

    def _parse_output(self, text: str):
        """
        解析LLM的输出，提取Thought和Action。
        """
        # Thought：匹配到 Action: 行或文本末尾
        thought_match = re.search(
            r"Thought:\s*(.*?)(?=\n\s*Action:|$)", text, re.DOTALL
        )
        # Action 只取当前这一行，避免把后面补充说明一起吞进来
        action_match = re.search(r"^\s*Action:\s*(.+)$", text, re.MULTILINE)
        thought = thought_match.group(1).strip() if thought_match else None
        action = self._clean_action(action_match.group(1)) if action_match else None
        return thought, action

    def _parse_action(self, action_text: str):
        """
        解析Action字符串，提取工具名和输入。
        """
        action = self._clean_action(action_text)
        match = re.search(r"(\w+)\[(.*)\]", action, re.DOTALL)
        if match:
            return match.group(1), match.group(2).strip()
        return None, None


if __name__ == "__main__":
    llm = HelloAgentsLLM()
    tool_executor = ToolExecutor()
    search_desc = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    tool_executor.registerTool("Search", search_desc, search)
    agent = ReActAgent(llm_client=llm, tool_executor=tool_executor)
    question = "华为最新的手机是哪一款？它的主要卖点是什么？"
    agent.run(question)
