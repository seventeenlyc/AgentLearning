import json
import os
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(self,path:str="",
                 raw_memory:List[Dict] | None = None,
                 model=None,
                 ):
        self.path = path
        self.model = model or SentenceTransformer("BAAI/bge-small-zh-v1.5")
        self.memory=[]
        self._load_memory(path,raw_memory)
        self.documents=[self._build_search_text(x) for x in self.memory]
        self.embeddings=self.model.encode(self.documents,normalize_embeddings=True)


    def build_search_text(self,memory:Dict):
        return self._build_search_text(memory)


    def add_memory(self,memory:Dict):
        return self._add_memory(memory)


    def search(self,query:str,top_k:int,min_score=0.35):
        return self._search(query,top_k,min_score)



    def _build_search_text(self,memory:Dict):
        try:
            return (
            f"类型：{memory['type']}\n"
            f"摘要：{memory['summary']}\n"
            f"标签：{','.join(memory['tags'])}\n"
            f"相关实体：{','.join(memory['entities'])}\n"
            f"时间：{memory.get('event_time', '')}\n"
            f"地点：{memory.get('location', '')}\n"
            f"内容：{memory['content']}"
            )
        except Exception:
            raise self._invalid_memory()


    def _load_memory(self,path,raw_memory):
        if path:
            if not os.path.exists(path):
                raise FileNotFoundError(f"记忆文件不存在: {os.path.abspath(path)}")
            with open(path,'r',encoding="utf-8") as f:
                memory = json.load(f)
            self.memory.extend(memory)
        if raw_memory:
            self.memory.extend(raw_memory)


    @staticmethod
    def _check_memory(memory:Dict):
        must_element=["type","summary","tags","entities","event_time","location","content"]
        return all(must in memory.keys() for must in must_element)


    def _add_memory(self,memory:Dict):
        if self._check_memory(memory):
            self.memory.append(memory)
            self._add_embedding(memory)
            self._save_memory()
        else:
            raise self._invalid_memory()


    def _add_document(self,memory:Dict):
        self.documents.append(self.build_search_text(memory))
        return self.build_search_text(memory)


    def _add_embedding(self,memory:Dict):
        new_embedding=self.model.encode(self._add_document(memory),normalize_embeddings=True)
        self.embeddings=np.vstack([self.embeddings,new_embedding])



    def _search(self,query:str,top_k=3,min_score=0.35):
        query_embedding=self.model.encode(query,normalize_embeddings=True)
        scores=self.embeddings @ query_embedding
        top_vector=np.argsort(scores)[::-1][:top_k]
        results=[]
        for i in top_vector:
            score=scores[i]
            if score<min_score:
                continue
            results.append({
                "memory":self.memory[i],
                "score":float(score),
            })
        return results


    def _save_memory(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.memory, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _invalid_memory():
        return ValueError("memory格式错误，参考格式："
               """{
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
                """
               )