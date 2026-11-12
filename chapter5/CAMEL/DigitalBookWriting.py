import os

from colorama import Fore
from camel.models import ModelFactory
from camel.societies import RolePlaying
from camel.types import ModelPlatformType
from camel.utils import print_text_animated
from dotenv import load_dotenv

load_dotenv()

# 模型服务二选一（.env 里用 LLM_PROVIDER 切换）：
#   zhipu（默认）-> 智谱 GLM，复用 chapter4 的 LLM_* 三个变量
#   qwen         -> 阿里百炼通义千问，用 DASHSCOPE_API_KEY 等
# 两套服务的密钥、地址、模型名不能混用。
# 阿里云百炼账户欠费 / 免费额度耗尽时，qwen 会返回 400 overdue-payment；
# 账户恢复后把 LLM_PROVIDER 改成 qwen 即可切回教材示例。
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "zhipu").strip().lower()

if LLM_PROVIDER == "qwen":
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 DASHSCOPE_API_KEY，请在 .env 中填入阿里百炼 API Key")
    model = ModelFactory.create(
        model_platform=ModelPlatformType.QWEN,
        model_type=os.getenv("QWEN_MODEL", "qwen-plus"),
        url=os.getenv(
            "QWEN_API_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        api_key=api_key,
    )
elif LLM_PROVIDER == "zhipu":
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 LLM_API_KEY，请在 .env 中填入智谱 API Key")
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
        model_type=os.getenv("LLM_MODEL_ID", "glm-4-flash"),
        url=os.getenv("LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
        api_key=api_key,
    )
else:
    raise ValueError(f"未知的 LLM_PROVIDER={LLM_PROVIDER!r}，只能是 qwen 或 zhipu")

print(Fore.YELLOW + f"当前模型服务: {LLM_PROVIDER}")

# 定义协作任务
task_prompt = """
创作一本关于"拖延心理学"的中文短篇电子书，目标读者是对心理学感兴趣的普通大众。
要求：
1. 内容科学严谨，基于实证研究
2. 语言通俗易懂，避免过多专业术语
3. 包含实用的改善建议和案例分析
4. 篇幅控制在8000-10000字
5. 结构清晰、包含引言、核心章节和总结
"""

print(Fore.YELLOW + f"协作任务:\n{task_prompt}\n")

# 初始化角色扮演绘画
# AI 作家作为"user"，负责提出写作结构和要求
# AI 心理学家作为"assistant"，负责提供专业知识和内容
role_play_session = RolePlaying(
    assistant_role_name="心理学家",
    user_role_name="作家",
    task_prompt=task_prompt,
    model=model,
    with_task_specify=False,  # 在本例中，我们直接使用给定的task_prompt
)

print(Fore.CYAN + f"具体任务描述:\n{role_play_session.task_prompt}\n")

# 开始协作对话
chat_turn_limit, n = 30, 0
# 调用 init_chat() 来获得由 AI 生成的初始对话消息
input_msg = role_play_session.init_chat()

while n < chat_turn_limit:
    n += 1
    # step() 方法驱动一轮完整的对话，AI 用户和 AI 助理各发言一次
    assistant_response, user_response = role_play_session.step(input_msg)

    # 检查是否有消息返回，防止对话提前终止
    if assistant_response.msg is None or user_response.msg is None:
        break

    print_text_animated(
        Fore.BLUE + f"作家（AI User）:\n\n{user_response.msg.content}\n"
    )
    print_text_animated(
        Fore.GREEN + f"心理学家（AI Assistant）:\n\n{assistant_response.msg.content}\n"
    )

    # 检查任务完成标志
    if (
        "<CAMEL_TASK_DONE>" in user_response.msg.content
        or "<CAMEL_TASK_DONE>" in assistant_response.msg.content
    ):
        print(Fore.MAGENTA + "🎉 电子书创作完成！")
        break

    # 将助理的回复作为下一轮对话的输入
    input_msg = assistant_response.msg

print(Fore.YELLOW + f"总共进行了 {n} 轮协作对话")
