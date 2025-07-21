from langchain_openai import ChatOpenAI      # pip install -U langchain-openai
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(
    model="model_name",                      # ← 你们的模型名
    base_url="http://your-company-llm-endpoint/v2",   # 指到 “/v2”
    api_key="",                              # 无鉴权就留空
    temperature=0.7,
)

# 单轮对话：直接传消息列表
reply = llm.invoke([HumanMessage(content="写一首关于 LangChain 的五言绝句")])
print(reply.content)



from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}")
])

chain = prompt | llm            # Runnable 拼接
print(chain.invoke({"question": "用三句话介绍机器学习"}).content)





# agent

from langchain.tools import Tool

def search_docs(query: str) -> str:
    """这里随便写个示例函数，你可以接公司内搜索接口"""
    return f"（模拟搜索结果）关于『{query}』的 Top‑3 文档"

tools = [
    Tool(
        name="SearchDocs",
        func=search_docs,
        description="当需要查阅知识库时使用这个工具。输入应是要搜索的关键词"
    )
]
from langchain.agents import initialize_agent, AgentType

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,   # 经典 ReAct
    verbose=True,                                  # 打印推理过程
)
question = "请问 LangChain 是什么？"
answer = agent.run(question)
print("最终回答：\n", answer)






# agent B

from langchain.tools import tool
from langchain_core.documents import Document

@tool
def fetch_weather(city: str) -> str:
    """查询指定城市的实时天气。"""
    # 这里随意返回
    return f"{city} 25℃，多云"

from langchain.agents import create_openai_functions_agent, AgentExecutor

agent = create_openai_functions_agent(
    llm=llm,
    tools=[fetch_weather],
    system_message="你是一位多功能助手，可以查询天气、搜索文档等。",
)

agent_executor = AgentExecutor(agent=agent, tools=[fetch_weather], verbose=True)
resp = agent_executor.invoke({"input": "帮我查一下上海今天的天气，然后用一句诗来形容"})
print(resp["output"])
