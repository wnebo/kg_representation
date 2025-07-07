from semantic_kernel.connectors.ai.ai_service_client_base import AIServiceClientBase
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.chat_completion import ChatRequestSettings
from semantic_kernel.orchestration.sk_context import SKContext

import requests
import json
from typing import List
from semantic_kernel.connectors.ai.chat_completion import ChatMessageContent, ChatRole

class MyCompanyChatCompletion(ChatCompletionClientBase):
    def __init__(self, endpoint: str, model: str):
        self.endpoint = endpoint
        self.model = model

    async def complete_chat_async(
        self,
        messages: List[ChatMessageContent],
        settings: ChatRequestSettings,
        context: SKContext = None,
    ) -> ChatMessageContent:
        # 构造 prompt 格式
        msgs = [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
        ]

        data = {
            "model": self.model,
            "messages": msgs
        }

        headers = {"Content-Type": "application/json"}

        response = requests.post(
            self.endpoint,
            headers=headers,
            data=json.dumps(data),
            proxies={'http': None, 'https': None}
        )

        response_data = response.json()
        reply = response_data['choices'][0]['message']['content']

        return ChatMessageContent(
            role=ChatRole.ASSISTANT,
            content=reply
        )

from semantic_kernel import Kernel

kernel = Kernel()

# 注册你自定义的 LLM 作为服务
chat_service = MyCompanyChatCompletion(
    endpoint="http://your-company-llm-endpoint",
    model="your-model-name"
)
kernel.add_chat_service("company-llm", chat_service)

# 加载一个简单的聊天技能
import semantic_kernel.skill_definition as sk_func

class MyChatSkill:
    @sk_func.kernel_function(name="chat")
    def chat(self, input: str) -> str:
        return f"User said: {input}"

kernel.import_skill(MyChatSkill(), skill_name="chat_skill")

# 执行一次调用
chat_history = []
chat_input = "你好，帮我写一个寒假计划"
settings = ChatRequestSettings()

reply = await chat_service.complete_chat_async(
    messages=[
        ChatMessageContent(role=ChatRole.USER, content=chat_input)
    ],
    settings=settings
)

print("🤖 LLM 回复：", reply.content)















import requests, json
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import OpenAIChatCompletion  # 参考现有实现
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_prompt import OpenAIChatPromptSettings
from semantic_kernel.connectors.ai.base import ChatCompletionClient  # 基类路径

class MyCompanyChatCompletion(ChatCompletionClient):
    def __init__(self, endpoint: str, model: str):
        self.endpoint = endpoint
        self.model = model

    def create_chat_completion(self, prompt_messages: list, **kwargs):
        # prompt_messages: list of dicts {"role":"user"/"assistant", "content": "..."}
        data = {"model": self.model, "messages": prompt_messages}
        headers = {"Content-Type": "application/json"}
        resp = requests.post(self.endpoint, headers=headers, data=json.dumps(data), proxies={'http':None,'https':None})
        resp.raise_for_status()
        j = resp.json()
        return j["choices"][0]["message"]

from semantic_kernel import Kernel

kernel = Kernel()
svc = MyCompanyChatCompletion(endpoint="http://your-endpoint", model="model_name")
kernel.register_chat_completion_service("company-llm", svc)

agent = ChatCompletionAgent(
    service=svc,
    name="MyCompanyAgent",
    instructions="你是一个智能中文助手。"
)

response = agent.get_response(messages=[{"role": "user", "content": "你好"}])
print(response.content)









import json
import requests
from typing import Any, Optional, List
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.chat_message_content import ChatMessageContent

class MySimpleLLMClient(ChatCompletionClientBase):
    def __init__(self, model_id: str, endpoint: str):
        self._model_id = model_id
        self._endpoint = endpoint

    @property
    def ai_model_id(self) -> str:
        return self._model_id

    async def get_chat_message_content(
        self,
        chat_history: ChatHistory,
        settings: Optional[PromptExecutionSettings] = None,
        **kwargs: Any,
    ) -> ChatMessageContent:
        messages = [
            {"role": msg.role.value, "content": msg.content}
            for msg in chat_history.messages
        ]
        payload = {
            "model": self._model_id,
            "messages": messages,
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            self._endpoint,
            headers=headers,
            data=json.dumps(payload),
            proxies={"http": None, "https": None}
        )
        response.raise_for_status()
        response_data = response.json()
        content = response_data["choices"][0]["message"]["content"]
        return ChatMessageContent(role="assistant", content=content)
import asyncio
from semantic_kernel.kernel import Kernel
from semantic_kernel.contents.chat_history import ChatHistory

async def main():
    kernel = Kernel()
    client = MySimpleLLMClient(
        model_id="your-model-name",
        endpoint="http://your-company-endpoint"
    )
    kernel.add_service(client)

    chat = ChatHistory()
    chat.add_user_message("你好，可以帮我写一个寒假学习计划吗？")
    result = await client.get_chat_message_content(chat)
    print("LLM 回复：", result.content)

asyncio.run(main())
