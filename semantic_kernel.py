import asyncio
import httpx
from openai import AsyncOpenAI
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

# ① 关闭代理（如果你在公司内网） + 指定 base_url
http_client = httpx.AsyncClient(proxies=None, timeout=30)

custom_client = AsyncOpenAI(
    base_url="http://your-company-llm-endpoint/v1",  # ← 只改这一行
    api_key="",                                      # 无鉴权留空
    http_client=http_client,
)

# ② 用这个 client 实例化 SK 的 OpenAIChatCompletion
service = OpenAIChatCompletion(
    ai_model_id="model_name",        # 对应你们接口里的 "model"
    async_client=custom_client,      # 关键参数
)

# ③ 像官方示例一样创建 Agent
agent = ChatCompletionAgent(
    service=service,
    name="SK-Assistant",
    instructions="You are a helpful assistant."
)

async def main():
    rsp = await agent.get_response(messages="写一首关于 Semantic Kernel 的五言绝句")
    print(rsp.content)

asyncio.run(main())










from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings
from semantic_kernel.functions import KernelArguments
from semantic_kernel.agents import ChatCompletionAgent

# 1) 构造 PromptExecutionSettings
settings = OpenAIChatPromptExecutionSettings(
    service_id="your-service-id",   # 和你注册的 service_id 保持一致
    ai_model_id="model_name",       # 你要用的模型
    temperature=0.7,                # 调整温度
    max_tokens=800,                 # 最大 token 数
    top_p=0.9,                      # nucleus 采样
    stream=False                    # 是否开启流式
)

# （可选）如果你有插件要自动调用，也可以在这里打开它：
# from semantic_kernel.connectors.ai.prompt_execution_settings import ToolCallBehavior
# settings.tool_call_behavior = ToolCallBehavior.AutoInvokeKernelFunctions()

# 2) 包装成 KernelArguments
args = KernelArguments(settings)

# 3) 把 arguments 传给你的 Agent
agent = ChatCompletionAgent(
    service=service,
    name="SK-Assistant",
    instructions="You are a helpful waiter.",
    plugins=[MenuPlugin()],
    arguments=args            # ← 这里传入的 settings 将作用于所有 get_response 调用
)

