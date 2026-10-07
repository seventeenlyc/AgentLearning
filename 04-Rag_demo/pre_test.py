import json
import os

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()
prompt="""你是一个基于检索资料回答问题的助手。

请只根据以下资料回答用户问题。
如果资料不足以回答，请明确说“根据现有资料无法确定”。

用户问题：
{query}

检索资料：
{retrieve}
"""
llm=OpenAI(base_url=os.getenv("ASTRAFLOW_BASE_URL"),api_key=os.getenv("ASTRAFLOW_API_KEY"))

def build_search_text(memory):
    return (
        f"类型：{memory['type']}\n"
        f"摘要：{memory['summary']}\n"
        f"标签：{','.join(memory['tags'])}\n"
        f"相关实体：{','.join(memory['entities'])}\n"
        f"时间：{memory.get('event_time', '')}\n"
        f"地点：{memory.get('location', '')}\n"
        f"内容：{memory['content']}"
    )


model=SentenceTransformer("BAAI/bge-small-zh-v1.5")
with open("data/memories.json","r",encoding="utf-8") as f:
    memories=json.load(f)
documents=[build_search_text(memory) for memory in memories]

documents_embedding=model.encode(documents,normalize_embeddings=True)


def search(query,top_k=3):
    query_embedding=model.encode(query,normalize_embeddings=True)
    scores=documents_embedding @ query_embedding
    top_vector=np.argsort(scores)[::-1][:top_k]
    results=[]
    for i in top_vector:
        results.append({
            "memory":memories[i],
            "score":float(scores[i])
        })
    return results

with open("data/question.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

hit1_count=0
hit3_count = 0

for question in questions:

    results = search(
        question["question"],
        top_k=5
    )

    retrieved_ids = {
        result["memory"]["id"]:result["score"]
        for result in results
    }
    retrieved_content={result["memory"]["id"]:result["memory"]["content"] for result in results}
    expected = question["expected"]

    expected_set = set(question["expected"])
    retrieved_set = set(retrieved_ids.keys())

    recall_at_3 = (
            len(expected_set & retrieved_set)
            / len(expected_set)
    )
    hit3=any(memory_id in retrieved_ids for memory_id in expected)
    hit1 = results[0]["memory"]["id"] == expected[0] if len(expected)==1 else all(memory_id in retrieved_ids for memory_id in expected)
    if hit3:
        hit3_count += 1

    if hit1:
        hit1_count += 1

    message=[{"role":"user","content":prompt.format(query=question["question"],retrieve=retrieved_content)}]
    print("=" * 50)
    print("问题：", question["question"])
    print("正确：", expected)
    print("检索：", retrieved_ids)
    print("Recall@5：", recall_at_3)
    print("Hit@1", hit1)
    print("Hit@5", hit3)
    response=llm.chat.completions.create(model=os.getenv("ASTRAFLOW_MODEL"),
                                         messages=message,temperature=0.1)
    print(response.choices[0].message.content)

accuracy = hit3_count / len(questions)

print()
print(f"最终 Hit@5：{accuracy:.2%}")
print(f"Hit@1概率: {hit1_count / len(questions):.2%}")