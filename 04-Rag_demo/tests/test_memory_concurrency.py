import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path
from threading import Event

import numpy as np
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

RAG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAG_ROOT))

from Agent.memory_agent import memory_agent
from retriever import Retriever


def agent_config():
    # 主入口包含交互循环，仅读取真实配置，避免启动模型或读取个人数据。
    tree = ast.parse((RAG_ROOT / "Agent" / "main.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "config"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("主入口缺少 config")


class Encoder:
    def __init__(self, deletion_started):
        self.deletion_started = deletion_started

    def encode(self, texts, normalize_embeddings=True):
        if isinstance(texts, str):
            # 放大更新与删除交错的窗口；顺序执行时超时后正常完成更新。
            if "updated" in texts:
                self.deletion_started.wait(timeout=0.5)
            values = np.array([len(texts) + 1, sum(map(ord, texts)) % 997 + 1], dtype=float)
            return values / np.linalg.norm(values)
        return np.array([self.encode(text) for text in texts]).reshape(-1, 2)


class Model:
    model_name = "offline-regression"

    def bind_tools(self, tools):
        self.tools = {item.name: item for item in tools}
        return self

    def invoke(self, messages):
        return AIMessage(content="done")


class MemoryConcurrencyTests(unittest.TestCase):
    def test_parallel_delete_and_revision_preserve_unrelated_record(self):
        config = agent_config()
        deletion_started = Event()
        semantic = dict(
            content="Example", type="event", summary="Example", tags=[],
            entities=[], event_time=None, location=None,
        )
        memories = [dict(id=key, **semantic, metadata={}) for key in ("a", "b", "c")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memories.json"
            path.write_text(json.dumps(memories), encoding="utf-8")
            ret = Retriever(path=str(path), model=Encoder(deletion_started))
            model = Model()
            _, operation = memory_agent(ret, model, InMemorySaver(), config)
            operation.invoke({"task": "inspect", "recent_context": "fixture"})

            # 保留实际删除逻辑，只用事件告知编码器另一个工具已开始执行。
            original_delete = ret.delete_memory

            def delete(memory_id):
                result = original_delete(memory_id)
                deletion_started.set()
                return result

            ret.delete_memory = delete
            calls = AIMessage(content="", tool_calls=[
                {"name": "revise_memory", "id": "revise-b", "args": {
                    "memory_id": "b", "semantic": dict(semantic, content="updated"),
                }},
                {"name": "delete_memory", "id": "delete-a", "args": {"memory_id": "a"}},
            ])
            graph = StateGraph(MessagesState)
            graph.add_node("tools", ToolNode(list(model.tools.values())))
            graph.add_edge(START, "tools")
            graph.add_edge("tools", END)
            result = graph.compile().invoke({"messages": [calls]}, config=config)
            self.assertTrue(all(item.status == "success" for item in result["messages"][1:]))
            self.assertEqual([item["id"] for item in ret.memory], ["b", "c"])
            self.assertEqual(ret.memory[0]["content"], "updated")
            self.assertEqual(ret.memory[1]["content"], "Example")
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), ret.memory)
            self.assertEqual(len(ret.documents), len(ret.memory))
            self.assertEqual(len(ret.embeddings), len(ret.memory))
            for index, document in enumerate(ret.documents):
                np.testing.assert_allclose(ret.embeddings[index], ret.model.encode(document))


if __name__ == "__main__":
    unittest.main()
