from ast import literal_eval
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.graph import StateGraph, MessagesState, START, END
import os
import requests
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
load_dotenv()


@tool
def calc(a: int | float, b: int | float, op: str) -> int | float | None:
    """计算器，支持加减乘除、乘方（**）、取模（%）运算"""
    if op == "+":
        return a + b
    elif op == "-":
        return a - b
    elif op == "*":
        return a * b
    elif op == "/":
        return a / b
    elif op == "%":
        return a % b
    elif op == "**":
        return a ** b
    else:
        return None


@tool
def get_weather(city:str):
    """
    通过调用 wttr.in API 查询真实的天气信息。
    """
    url = f"https://wttr.in/{city}?format=j1"

    try:
        #发起网络请求
        response=requests.get(url)
        #检查是否状态码为200
        response.raise_for_status()
        data=response.json()
        current_condiction=data['current_condition'][0]
        weatherDesc=current_condiction["weatherDesc"][0]["value"]
        temp_C=current_condiction["temp_C"]
        return f"{city}当前天气：{weatherDesc}，当前温度：{temp_C}"
    except requests.exceptions.RequestException as e:
        return f"错误：网络问题{e}"
    except (KeyError, IndexError) as e:
        return f"错误：解析天气数据失败，可能是城市输入错误"





tools=[calc,get_weather]

model = ChatOpenAI(model=os.getenv("ASTRAFLOW_MODEL"),
                   api_key=os.getenv("ASTRAFLOW_API_KEY"),
                   base_url=os.getenv("ASTRAFLOW_BASE_URL"), )

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

checkpoint=InMemorySaver()
app1=g.compile(checkpointer=checkpoint)
app2=g.compile(checkpointer=checkpoint)
inputs = lambda question: {
    "messages": [
        {
            "role": "user",
            "content": question
        }
    ]
}
config = {
    "configurable": {
        "thread_id": "user_001"
    }
}
res1=app1.invoke(inputs("查询哈尔滨的天气"),config=config)

res1["messages"][-1].pretty_print()

res2=app2.invoke(inputs("温度转化成开尔文"),config=config)

res2["messages"][-1].pretty_print()
# 查看当前 thread 最新状态
snapshot = app1.get_state(config)

print("\n\n\n\n当前 State：")
for msg in snapshot.values["messages"]:
    msg.pretty_print()


history = app1.get_state_history(config)

for i, snapshot in enumerate(history):
    print(f"\n===== Checkpoint {i} =====")
    print("next:", snapshot.next)

    for msg in snapshot.values["messages"]:
        print(msg.type, ":", msg.content)