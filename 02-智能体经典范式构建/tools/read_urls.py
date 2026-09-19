import requests
from bs4 import BeautifulSoup
import re
from ast import literal_eval
from Memory import *
from typing import List,Dict
from requests.exceptions import (
    HTTPError,
    Timeout,
    ConnectionError,
    SSLError,
    RequestException
)
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
    error=[]
    if tool_input.get('urls') is None:
        print("请传入需要搜索的urls")
        return ["错误！缺少字段urls，请检查tool_input"],"错误！缺少字段urls，请检查tool_input"
    elif tool_input.get("question") is None:
        print("请传入question字段")
        return ["错误！缺少字段question，请检查tool_input"],"错误！缺少字段question，请检查tool_input"

    cleaned_urls =[clean_url(url) for url in tool_input.get("urls")]
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    observation=[]
    for url in cleaned_urls:
        for i in range(1,4):
            success=False
            try:
                response = requests.get(url, headers=headers,timeout=i*3+10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'lxml')
                for tag in soup([
                    "script", "style","nav","footer","header","noscript","svg"
                ]):
                    tag.decompose()
                content=soup.find("article") or  soup.find("main") or soup.body

                if content is None:
                    observation.append(f"URL: {url}\n""读取成功，但没有找到正文区域。")
                    break
                raw_text=content.get_text(separator='\n',strip=True)
                if not raw_text:
                    observation.append(f"URL: {url}\n""网页正文为空。")
                    error.append({url: "网页正文为空"})
                    break
                observation.append(f"来源URL{url},正文：\n{raw_text}")
                success=True
                break

            except HTTPError as e:
                status_code=(e.response.status_code if e.response is not None else None)
                if status_code in (400,401,403,404):
                    print(f"网页返回 {status_code}，"f"不再重试：{url}")
                    observation.append( f"URL: {url}\n" f"读取失败：HTTP {status_code}")
                    error.append({url:status_code})
                    break

                if status_code is not None and status_code>=500:
                    print(f"第 {i} 次读取失败："f"HTTP {status_code}")
                    if i==3:
                        observation.append(f"URL: {url}\n"f"连续3次读取失败："f"HTTP {status_code}")
                        error.append({url:status_code})
                        continue
                observation.append( f"URL: {url}\n" f"HTTP错误：{e}" )
                break
            except Timeout as e:
                print(f"{url}：第{i}次读取超时")
                if i==3:
                    print(f"{url}:三次读取均超时")
                    success_read[-1]['reason'] = 'Timeout'
            except (ConnectionError, SSLError) as e:
                print(f"第 {i} 次连接失败："f"{e}")
                if i == 3:
                    observation.append(f"URL: {url}\n"f"连接失败：{e}")
                    error.append({url:e})

            except RequestException as e:
                observation.append(f"URL: {url}\n"f"请求异常：{e}")
                error.append({url:e})
                break

            except Exception as e:
                observation.append(f"URL: {url}\n" f"网页处理异常：{e}")
                error.append({url:e})
                break

            if success:
                print(f"读取成功：{url}")
        else:
            print("当前读取连续失败 3 次")

    if not observation:
        return error,"所有URL均读取失败，没有获得可用正文信息。"
    return error,"\n\n".join(observation)



func=read_urls