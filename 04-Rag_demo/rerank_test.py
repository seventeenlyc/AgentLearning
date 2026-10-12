import json
from pathlib import Path

from sentence_transformers import CrossEncoder

from rag import RAG
from retriever import Retriever


def evaluate_hits(results, expected):
    expected_ids = set(expected or [])
    retrieved_ids = [item["memory"]["id"] for item in results]
    # Hit@K：前 K 条中至少有一条正确记忆，统一使用完整 ID 匹配。
    hit1 = bool(expected_ids.intersection(retrieved_ids[:1]))
    hit3 = bool(expected_ids.intersection(retrieved_ids[:3]))
    return hit1, hit3


if __name__ == '__main__':
    data_dir = Path(__file__).resolve().parent / "data"
    with (data_dir / "question.json").open("r", encoding="utf-8") as f:
        questions = json.load(f)
    if not questions:
        raise ValueError("question.json 中没有评测问题")

    ret=Retriever(path=str(data_dir / "memories.json"))
    reranker=CrossEncoder("BAAI/bge-reranker-base")
    rag=RAG(ret,reranker)

    rerank_top3=0
    rerank_top1=0
    for question in questions:
        query=question["question"]
        expected=question["expected"]
        rerank_results=rag.retrieve_and_rerank(query,recall_k=15,final_k=5,min_score=0.45)
        hit1,hit3=evaluate_hits(rerank_results,expected)
        if hit1:
            rerank_top1+=1
        if hit3:
            rerank_top3+=1

        print("=" * 50)
        print("问题：", query)
        print("正确：", expected)
        print("检索：", [r["memory"]["id"] for r in rerank_results])
        if expected:
            print("Hit@1", hit1)
            print("Hit@3", hit3)
        else:
            if not rerank_results:
                rerank_top3+=1
                rerank_top1+=1
                print("无答案，检索结果与实际相符")
    print()
    print(f"最终 Hit@3：{rerank_top3 / len(questions):.2%}")
    print(f"Hit@1概率: {rerank_top1 / len(questions):.2%}")

