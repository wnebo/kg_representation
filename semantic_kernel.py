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
