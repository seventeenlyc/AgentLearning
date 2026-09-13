import requests
from bs4 import BeautifulSoup
import re
from ast import literal_eval
from typing import List,Dict
from Compacter import Compacter
from HelloAgentLLM import HelloAgentLLM
name="read_urls"
description = """
读取指定网页的正文内容。
tool_input={
    "urls":[
    "https://example.com","https://example2.com"
    ],
    "question":"需要获取到怎么样的信息"
}
"""
compact=True
judge_prompt="""
你是一个评判信息的专家，你需要根据已有的信息，判断是否已经收集到充足的信息回答用户的问题。
已有的信息：
{observation}

用户的问题：
{question}

你的输出只能在True和False里选择，不需要添加任何其他的语句。
示例输出：
True

"""
def clean_url(url) -> str:
    url = url.strip()
    match = re.fullmatch(
        r"\[.*?]\((https?://.*?)\)",
        url
    )
    if match:
        return match.group(1)
    # 有时候模型可能自己包了一层 []
    if (
        url.startswith("[")
        and url.endswith("]")
    ):
        url = url[1:-1]

    return url.strip()


def read_urls(tool_input:Dict):
    if tool_input.get('urls') is None:
        print("请传入需要搜索的urls")
        return "错误！缺少字段urls，请检查tool_input"
    elif tool_input.get("question") is None:
        print("请传入question字段")
        return "错误！缺少字段question，请检查tool_input"
    try:
        compacter=Compacter()
        judger=HelloAgentLLM()
        cleaned_urls =[clean_url(url) for url in tool_input.get("urls")]
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        observation=[]
        for url in cleaned_urls:
            response = requests.get(url, headers=headers,timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'lxml')
            for tag in soup([
                "script", "style","nav","footer","header","noscript","svg"
            ]):
                tag.decompose()
            content=soup.find("article") or  soup.find("main") or soup.body

            if content is None:
                return "网页读取成功，但没有发现有效正文。"
            raw_text=content.get_text(separator='\n',strip=True)
            text=compacter.compact_message([raw_text],tool_input.get("question"))
            observation.append(text)
            reply=judger.generate([{"role":"user","content":judge_prompt.format(observation=observation,question=tool_input.get("question"))}],output=False)
            judge=literal_eval(reply)
            if judge:
                ans=compacter.compact_message(observation,tool_input.get("question"))
                print(ans)
                return ans

            observation.append("当前未能搜到完全足够的信息回答此问题，但请根据已有的信息进行总结。并在总结的结果里说明还缺少哪些信息。")
            ans=compacter.compact_message(observation,tool_input.get("question"))
            print(ans)
            return ans


    except Exception as e:
        return f"解析网页错误，{type(e).__name__}:{e}"
    return None


func=read_urls