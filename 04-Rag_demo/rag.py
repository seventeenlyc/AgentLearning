import os

from dotenv import load_dotenv
from openai import OpenAI

from retriever import Retriever

load_dotenv()
sys_prompt="""你是一个基于检索资料回答问题的助手。

请只根据以下资料回答用户问题，并且标注回答的来源。
如果资料不足以回答，请明确说“根据现有资料无法确定”。
"""
user_prompt="""
用户问题：
{query}

检索资料：
{retrieve}
"""


def _get_message(query:str, retrieve:str):
    return [{
        "role":"system",
        "content":sys_prompt,
    },
        {
        "role":"user",
        "content":user_prompt.format(query=query,retrieve=retrieve)
    }]

class RAG:
    def __init__(
        self,
        retriever:Retriever,
        model: str,
        base_url: str = None,
        api_key: str = None
    ):
        self.retriever = retriever

        self.client = OpenAI(
            api_key=api_key or os.getenv("ASTRAFLOW_API_KEY"),
            base_url=base_url or os.getenv("ASTRAFLOW_BASE_URL")
        )

        self.model = model

    def answer(self, query: str, top_k: int = 5):
        results = self.retriever.search(
            query,
            top_k=top_k
        )

        context = self._build_context(results)

        return self._answer(
            query,
            context
        )

    def _build_context(self, results):
        context = []

        for result in results:
            memory = result["memory"]

            context.append(
                f"[{memory['id']}]\n"
                f"时间：{memory.get('event_time', '')}\n"
                f"地点：{memory.get('location', '')}\n"
                f"内容：{memory['content']}"
            )

        return "\n\n".join(context)


    def _answer(self,query:str,context):
        response=self.client.chat.completions.create(model=self.model,
                                                     messages=_get_message(query, context),
                                                     temperature=0.1)
        return response.choices[0].message.content


if __name__ == "__main__":
    retriever = Retriever(path="data/memories.json")
    rag=RAG(retriever=retriever,model=os.getenv("ASTRAFLOW_MODEL"))
    answer=rag.answer("拾叁和拾柒什么时候确认关系？")
    print(answer)