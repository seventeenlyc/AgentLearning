import requests
import time
from langchain_core.tools import tool

from retriever import Retriever

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

def build_memory_tools(ret:Retriever):
    @tool
    def search_RAG(query:str,top_k:int,min_score=0.35):
        """
        在长期记忆里搜寻信息
        :param min_score: 搜索结果里允许的最低相似度
        :param query: 搜索的问题
        :param top_k: 希望返回的最相关记忆数目
        :return:
        """
        return ret.search(query,top_k=top_k,min_score=min_score)

    @tool
    def add_memory(memory: dict):
        """
        向长期记忆中添加一条新记录，记录格式：
        {
            "id": "memory_01",
            "type": "event",
            "content": "...",
            "summary": "...",
            "tags": ["..."],
            "entities": ["..."],
            "event_time": "2025-02-05",
            "location": "韶关",
            "source": {
              "type": "legacy_import",
              "id": "memory_01"
            },
            "created_at": "2026-10-06",
            "updated_at": "2026-10-06",
            "metadata": {}
        }
        :param memory: 添加的记忆
        :return:
        """
        ret.add_memory(memory)
        return "记忆已添加"
    return search_RAG,add_memory

def get_time():
    """返回当前本地时间"""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())