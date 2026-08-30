import os
from tavily import TavilyClient

def get_attraction(city:str,weather:str):
    """
    根据城市和天气，搜索并返回优化的结果
    """
    apikey=os.environ.get("TAVILY_API_KEY")
    if not apikey:
        return f"错误：请先配置TARVILY_API_KEY"

    tavily=TavilyClient(apikey)

    query=f"'{city}'在'{weather}'天气下最值得去的旅游景点"

    try:
        response=tavily.search(query,search_depth='basic',include_answer=True)
        if response.get("answer"):
            return response["answer"]
        result=[]
        for r in response["results"]:
            result.append(f"-{r['title']}:{r['content']}")
        if not result:
            return "抱歉，没有找到相关的旅游景点推荐"

        return "根据搜索，找到以下信息：\n"+"\n".join(result)

    except Exception as e:
        return f"错误：执行搜索时出现问题-{e}"