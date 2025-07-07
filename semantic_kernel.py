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
