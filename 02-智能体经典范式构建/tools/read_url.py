import requests
from bs4 import BeautifulSoup
import re
name="read_url"
description = """
读取指定网页的正文内容。
传入参数：网页URL字符串。
输入必须是一个完整网页URL，例如：
`https://example.com/article`
"""
compact=True
def clean_url(url: str) -> str:
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


def read_url(url:str):
    try:
        url=clean_url(url)
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
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
        text=content.get_text(separator='\n',strip=True)
        return text
    except Exception as e:
        return f"解析网页错误，{type(e).__name__}:{e}"
func=read_url