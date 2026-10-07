from ast import literal_eval
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.graph import StateGraph, MessagesState, START, END
import os
import requests
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from datetime import date, datetime, timezone, timedelta
from typing import Literal
from uuid import uuid4
from Agent.memory_service import MemorySemantic,create_memory
from Agent.memory_agent import memory_agent
from Agent.tools import calc,get_weather,build_memory_tools,get_time
from retriever import Retriever
load_dotenv()

checkpointer = InMemorySaver()
config={
    "configurable": {
        "thread_id": "user_001"
    }
}

model = ChatOpenAI(model=os.getenv("ASTRAFLOW_MODEL"),
                   api_key=os.getenv("ASTRAFLOW_API_KEY"),
                   base_url=os.getenv("ASTRAFLOW_BASE_URL"), )

retriever = Retriever(path="../data/memories.json")
add_memory,memory_operation=memory_agent(ret=retriever,model=model,checkpointer=checkpointer,config=config)
tools=[calc,get_weather,memory_operation,get_time]

model_with_tools=model.bind_tools(tools)


def call_model(state: MessagesState):
    response = model_with_tools.invoke(state["messages"])
    return {
        "messages": [response],
    }

g=StateGraph(MessagesState)
g.add_node("model",call_model)
g.add_node("tools",ToolNode(tools))

g.add_edge(START,"model")
g.add_conditional_edges("model",tools_condition)
g.add_edge("tools","model")


app=g.compile(checkpointer=checkpointer)
inputs = lambda question: {
    "messages": [
        {
            "role": "user",
            "content": question
        }
    ]
}
# 使用未绑定工具的 model
extractor = model.with_structured_output(
    MemorySemantic,
    method="function_calling",
)
while True:
    question=input("输入问题：\n")
    if question.lower()=="exit" or question.lower()=="q":
        break
    result=app.invoke(inputs(question),config=config)

    for r in result["messages"]:
        r.pretty_print()
