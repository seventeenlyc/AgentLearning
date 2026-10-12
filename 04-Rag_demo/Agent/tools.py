import os
import requests
import time
from urllib.parse import urlsplit
from langchain_core.tools import tool
from requests import HTTPError
from serpapi import SerpApiClient

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
        return f"错误：解析天气数据失败，可能是城市输入错误\n{e}"

def build_memory_tools(ret:Retriever):
    @tool
    def search_RAG(query:str,recall_k:int=10,final_k:int=5,min_score:float=-1.0):
        """
        在长期记忆里搜寻信息
        :param query: 需要查找的信息
        :param recall_k: 希望召回的信息数量
        :param final_k: 希望得到的答案数量
        :param min_score: 检索的最小得分
        :return: 检索结果
        """
        return ret.reranked_search(query,recall_k=recall_k,final_k=final_k,min_score=min_score)

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

@tool
def get_time():
    """返回当前本地时间"""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

@tool("search_web")
def google_search(query: str, num_results: int = 5, domains: list[str] | None = None) -> dict:
    """
    通过 SerpApi 搜索 Google，返回标题、摘要、链接和日期（未知日期为 null）。
    :param query: 搜索词，支持 Google 搜索语法。
    :param num_results: 最多返回多少条结果，范围 1 到 10，默认 5。
    :param domains: 可选域名列表，例如 ["ai.google.dev", "blog.google"]。
        只保留这些域名及其子域名的链接，不返回未限定来源的答案框。
    """
    if not query.strip() or not 1 <= num_results <= 10:
        return {"status": "error", "query": query, "error": "搜索词不能为空，结果数量应为 1 到 10。"}

    allowed_domains = set()
    try:
        for domain in domains or []:
            host = urlsplit(domain if "://" in domain else f"//{domain}").hostname
            if not host:
                return {"status": "error", "query": query, "error": "请输入有效域名。"}
            allowed_domains.add(host.lower().rstrip("."))
    except ValueError:
        return {"status": "error", "query": query, "error": "请输入有效域名。"}

    api_key = os.getenv("SEARCH_API_KEY")
    if not api_key:
        return {"status": "error", "query": query, "error": "请先配置 SEARCH_API_KEY。"}
    params = {
        "q": query,
        "engine": "google",
        "api_key": api_key,
        "gl": "cn",  # 国家代码
        "hl": "zh-cn"  # 语言代码
    }
    try:
        client = SerpApiClient(params, engine="google", timeout=(5, 30))
        results = client.get_dict()
        if results.get("error"):
            return {"status": "error", "query": query,
                    "error": str(results["error"]).replace(api_key, "[redacted]")}

        candidates = []
        # 先检查整个结果列表中的域名，再截取数量，避免漏掉排位靠后的官方来源。
        for item in results.get("organic_results", []):
            url = item.get("link", "")
            try:
                parsed = urlsplit(url)
            except ValueError:
                continue
            host = (parsed.hostname or "").lower().rstrip(".")
            if parsed.scheme not in {"http", "https"} or not host:
                continue
            if allowed_domains and not any(
                host == domain or host.endswith("." + domain) for domain in allowed_domains
            ):
                continue
            candidates.append({
                "position": item.get("position"), "title": item.get("title"),
                "snippet": item.get("snippet"), "url": url, "date": item.get("date"),
            })

        information = results.get("search_information", {})
        answer_box = None if allowed_domains else results.get("answer_box")
        knowledge_graph = None if allowed_domains else results.get("knowledge_graph")
        return {
            "status": "ok" if candidates or answer_box or knowledge_graph else "no_results",
            "query": query,
            "query_displayed": information.get("query_displayed", query),
            "showing_results_for": information.get("showing_results_for"),
            "results": candidates[:num_results],
            "answer_box": answer_box,
            "knowledge_graph": knowledge_graph,
        }

    except requests.Timeout:
        return {"status": "error", "query": query, "error_type": "timeout",
                "error": "联网搜索超时，未获取到搜索结果。"}
    except requests.RequestException:
        return {"status": "error", "query": query, "error_type": "network",
                "error": "搜索服务连接失败，未获取到搜索结果。"}
    except ValueError:
        return {"status": "error", "query": query, "error_type": "invalid_response",
                "error": "搜索接口未返回有效的 JSON。"}


from langchain_openai import ChatOpenAI

@tool("gemini_search_web")
def gemini_google_search(query: str) -> str:
    """通过 Gemini 内置 Google 搜索查询互联网信息（Gemini模型专用）。"""
    search_model = ChatOpenAI(
        model=os.getenv("LLM_MODEL"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("BASE_URL"),
    ).bind(
        tools=[{"type": "google_search"}]
    )

    response = search_model.invoke(query)
    return str(response.text)


@tool
def read_url(url:str):
    """打开网页并读取正文"""
    from bs4 import BeautifulSoup
    import requests
    try:
        response = requests.get(url,headers={"User-Agent": "Mozilla/5.0"},timeout=(5,20))
        soup = BeautifulSoup(response.content, "lxml")
        for tag in soup([
            "script", "style", "nav", "footer", "header", "noscript", "svg"
        ]):
            tag.decompose()
        content = soup.find("article") or soup.find("main") or soup.body
        if not content:
            return f"访问{url}成功，但没有找到正文区域"
        raw_text = content.get_text(separator='\n', strip=True)
        if not raw_text:
            return f"{url}:网页正文为空"

        return raw_text
    except HTTPError as e:
        status_code = (e.response.status_code if e.response is not None else None)
        if status_code:
            raise HTTPError(status_code)
        else:
            raise HTTPError(e)