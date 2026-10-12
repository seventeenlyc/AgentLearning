import os

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import CrossEncoder
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
        reranker:CrossEncoder=None,
        model: str=None,
        base_url: str = None,
        api_key: str = None
    ):
        self.retriever = retriever
        self.reranker = reranker or CrossEncoder("BAAI/bge-reranker-base")
        self.client = OpenAI(
            api_key=api_key or os.getenv("ASTRAFLOW_API_KEY"),
            base_url=base_url or os.getenv("ASTRAFLOW_BASE_URL")
        )

        self.model = model

    def answer(self, query: str,*, recall_k:int =10,final_k: int = 5,min_score:float=-1.0):
        results = self._retrieve_and_rerank(
            query,
            recall_k=recall_k,
            final_k=final_k,
            min_score=min_score,
        )

        context = self._build_context(results)

        return self._answer(
            query,
            context
        )


    def retrieve_and_rerank(self,query: str,recall_k:int=10,final_k:int=5,min_score:float=-1.0):
        return self._retrieve_and_rerank(query,recall_k=recall_k,final_k=final_k,min_score=min_score)


    def _retrieve_and_rerank(self,query:str,recall_k:int=10,final_k:int=5,min_score:float=-1.0):
        candidates=self.retriever.raw_search(query,top_k=recall_k,min_score=min_score)
        if not candidates:
            return []
        pairs=[(query,self.retriever.build_search_text(candidate["memory"])) for candidate in candidates]
        scores=self.reranker.predict(pairs)
        results=[
            {
                "memory":item["memory"],
                "recall_score":item["score"],
                "rerank_score":float(score)
            }for item,score in zip(candidates,scores)
        ]
        results.sort(key=lambda item: item["rerank_score"],reverse=True)
        return results[:min(final_k,len(results))]

    def _build_context(self, results):
        context = []

        for result in results:
            memory = result["memory"]

            context.append(
            f"类型：{memory['type']}\n"
            f"摘要：{memory['summary']}\n"
            f"标签：{','.join(memory['tags'])}\n"
            f"相关实体：{','.join(memory['entities'])}\n"
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
