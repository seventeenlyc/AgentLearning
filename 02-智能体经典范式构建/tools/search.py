import os
from dotenv import load_dotenv
from serpapi import SerpApiClient
from typing import Dict
load_dotenv()
name="search"
description = """一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。
tool_input={"query":"你要搜索的问题"}
"""
compact=False
def search(query:Dict) ->str:
    """
    调用SerpApi进行网页搜索，优先返回直接答案或知识图谱信息。
    """
    if not query.get("query"):
        print("tool_input格式错误！请严格按照{\"query\":\"你要搜索的问题\"}的格式传参")
        return "tool_input格式错误，找不到query字段，请检查格式！"
    print(f"🔍 正在执行 [SerpApi] 网页搜索: {query.get('query')}")
    api_key=os.getenv("SEARCH_API_KEY")
    if not api_key:
        return f"错误：请先配置SerpApi的api_key"
    params={
        "q": query.get("query"),
        "engine":"google",
        "api_key":api_key,
        "gl":"cn",#国家代码
        "hl":"zh-cn"#语言代码
    }
    timeout=[0,0,15,20]
    try:
        for i in range(1,4):
            try:
                client = SerpApiClient(params,engine="google",timeout=timeout[i]+10)
                results=client.get_dict()

                if "answer_box_list" in results:
                    return "\n".join(results["answer_box_list"])
                if "answer_box" in results and "answer" in results["answer_box"]:
                    return results["answer_box"]["answer"]
                if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
                    return results["knowledge_graph"]["description"]
                if "organic_results" in results and results["organic_results"]:
                    # 如果没有直接答案，则返回前三个有机结果的摘要
                    snippets = [
                        f"[{i + 1}] {res.get('title', '')}\n摘要：{res.get('snippet', '')}\nURL:{res.get('link','')}"
                        for i, res in enumerate(results["organic_results"][:3])
                    ]
                    return "\n\n".join(snippets)


            except Exception as e:
                print(f"搜索时发生错误：{e}")

        return f"对不起，没有找到关于 '{query.get('query')}' 的信息。"

    finally:
        print("已退出搜索工具")



func=search