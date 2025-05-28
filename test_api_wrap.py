# my_llm_wrapper.py

from typing import Optional, List
from langchain_core.language_models.llms import LLM
import requests


class RagasJudgeLLM(LLM):
    """
    封装一个带 header 的简单 LLM API，兼容 LangChain 和 RAGAS。
    """
    url: str  # 你的 API 地址，比如 http://localhost:8000/generate

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        向 API 发送请求，提取 content 字段作为模型输出。
        """
        try:
            payload = {"prompt": prompt}
            headers = {
                "Content-Type": "application/json"  # 明确告诉服务器我们发送的是 JSON
            }

            response = requests.post(self.url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()

            content = data.get("content", "").strip()
            return content

        except Exception as e:
            raise ValueError(f"调用 LLM API 出错: {e}")
    async def _acall(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        异步调用：用于 RAGAS 或 LangChain 中的 async 流程。
        """
        try:
            payload = {"prompt": prompt}
            headers = {"Content-Type": "application/json"}
            async with aiohttp.ClientSession() as session:
                async with session.post(self.url, json=payload, headers=headers, timeout=30) as response:
                    if response.status != 200:
                        raise ValueError(f"HTTP错误: {response.status}")
                    data = await response.json()
                    content = data.get("content", "").strip()
                    return content
        except Exception as e:
            raise ValueError(f"调用 LLM API 异步出错: {e}")


    async def agenerate_prompt(self, prompts: List) -> dict:
        # 👇 手动把 PromptValue 转换成 str
        clean_prompts = [
            p.to_string() if hasattr(p, "to_string") else str(p)
            for p in prompts
        ]
        return await super().agenerate_prompt(clean_prompts)

    @property
    def _llm_type(self) -> str:
        return "ragas_judge_llm"
