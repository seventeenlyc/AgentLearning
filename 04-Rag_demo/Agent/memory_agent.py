from copy import deepcopy
from datetime import datetime, timezone, timedelta
from typing import Dict
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition

from Agent.memory_service import MemorySemantic
from Agent.memory_service import create_memory
from Agent.tools import build_memory_tools
from retriever import Retriever

load_dotenv()

def memory_agent(ret:Retriever,model:ChatOpenAI,checkpointer,config):
    """记忆agent，当需要修改记忆时调用"""
    @tool
    def add_memory(raw_content:str,semantic:MemorySemantic,auto_create:bool):
        """
        添加记忆，从用户原文提取记忆语义字段。
        :param auto_create: 是否由模型根据上下文自动创建，如果因为用户要求而直接/间接调用此工具，填False，否则为True
        :param raw_content: 用户原文
        :param semantic: content 是完整的规范化正文，summary 是简短摘要。
        依据原始消息时间解析相对日期，并与 event_time 保持一致。
        依据明确上下文解析人物指代。
        保留“下午”等细节，不得补充未知地点、人物或行为。
        无法确定的信息保持不确定，必要时请求澄清。格式如下：\n
        ```python\n
        class MemorySemantic(BaseModel):
            model_config = ConfigDict(extra="forbid")
            content: str = Field(
                min_length=1,
                description=(
                    "完整的规范化记忆正文，保留原文事实和细节。"
                    "有明确依据时，将相对日期转换为具体日期，"
                    "将代词替换为明确人物名称；不得补充未知事实。"
                ),
            )
            type: Literal[
                "event", "fact", "preference",
                "person", "plan", "opinion"
            ]
            summary: str = Field(min_length=1)
            tags: list[str]
            entities: list[str]
            event_time: date | None = Field(
                description=(
                    "事件日期。可以依据原始消息时间解析昨天、前天等表达；"
                    "缺少可靠时间依据时返回 null。"
                )
            )
            location: str | None = Field(
                description="原文有明确地点才填写，否则为 null"
            )
        ```

        :return: 生成的新记忆
        """
        if raw_content.startswith("/remember "):
            raw_content = raw_content[len("/remember "):]

        created_by={}
        if not auto_create:
            created_by = {
                "type": "user",
                "method": "manual_request",
            }
        else:
            created_by = {
                "type": "model",
                "method": "auto_extract"
            }
        memory = create_memory(
            ret,
            raw_content=raw_content,
            semantic=semantic.model_dump(mode="json"),
            source={
                "type": "conversation",
                "id": f"message_{uuid4().hex}",
            },
            created_by=created_by,
            processed_by={
                "type": "model",
                "name": model.model_name,
            },
        )
        ret.add_memory(memory)
        print(f"记忆已添加：{memory['id']}")
        return memory


    @tool
    def delete_memory(memory_id:str):
        """输入需要删除的记忆id，删除对应的记忆"""
        return ret.delete_memory(memory_id)


    @tool
    def revise_memory(memory_id:str,semantic:MemorySemantic):
        """
        修改记忆，从用户原文提取记忆语义字段。不得补充未提及的事实；日期或地点不明确时返回 null。
        :param memory_id:需要修改的记忆id
        :param semantic:content 是完整的规范化正文，summary 是简短摘要。
        依据原始消息时间解析相对日期，并与 event_time 保持一致。
        依据明确上下文解析人物指代。
        保留“下午”等细节，不得补充未知地点、人物或行为。
        无法确定的信息保持不确定，必要时请求澄清。格式如下：\n
        ```python\n
        class MemorySemantic(BaseModel):
            model_config = ConfigDict(extra="forbid")
            content: str = Field(
                min_length=1,
                description=(
                    "完整的规范化记忆正文，保留原文事实和细节。"
                    "有明确依据时，将相对日期转换为具体日期，"
                    "将代词替换为明确人物名称；不得补充未知事实。"
                ),
            )
            type: Literal[
                "event", "fact", "preference",
                "person", "plan", "opinion"
            ]
            summary: str = Field(min_length=1)
            tags: list[str]
            entities: list[str]
            event_time: date | None = Field(
                description=(
                    "事件日期。可以依据原始消息时间解析昨天、前天等表达；"
                    "缺少可靠时间依据时返回 null。"
                )
            )
            location: str | None = Field(
                description="原文有明确地点才填写，否则为 null"
            )
        ```
        :return:修改结果
        """
        def get_diff(old:Dict,new:Dict):
            changes=[]
            for item in new:
                if old.get(item) != new.get(item):
                    changes.append({
                        item:{
                            "operation":"replace",
                            "before":old[item],
                            "after":new[item]
                        }
                    })

            return changes

        old = next(
            (deepcopy(item) for item in ret.memory
             if item["id"] == memory_id),
            None,
        )
        if old is None:
            return {"success": False, "reason": "记忆不存在"}
        new_memory = deepcopy(old)
        new_memory.update(semantic.model_dump(mode="json"))

        changes = get_diff(old, new_memory)

        if not changes:
            return {
                "success": True,
                "changed": False,
                "memory_id": memory_id,
            }

        now = datetime.now(
            timezone(timedelta(hours=8))
        ).isoformat(timespec="seconds")

        new_memory["updated_at"] = now
        new_memory["processed_by"] = {
            "type": "model",
            "name": model.model_name,
        }

        metadata = new_memory.setdefault("metadata", {})
        metadata["version"] = metadata.get("version", 1) + 1
        metadata.setdefault("diff", []).append({
            "modified_at": now,
            "changes": changes,
        })

        result = ret.update_memory(memory_id, new_memory)

        if not result:
            return {"success": False, "reason": "记忆更新失败"}

        print(f"记忆已修改：{memory_id}")
        return {
            "success": True,
            "changed": True,
            "memory_id": memory_id,
        }


    @tool
    def memory_operation(task:str,*,recent_context):
        """处理长期记忆的增删改查任务。"""
        search_RAG=build_memory_tools(ret)[0]
        tools=[search_RAG,add_memory,delete_memory,revise_memory]
        model_with_tools=model.bind_tools(tools)
        def call_model(state: MessagesState):
            response = model_with_tools.invoke(state["messages"])
            return {
                "messages": [response]
            }
        m=StateGraph(MessagesState)
        m.add_node("model",call_model)
        m.add_node("tools",ToolNode(tools))
        m.add_edge(START,"model")
        m.add_conditional_edges("model",tools_condition)
        m.add_edge("tools","model")
        memo_agent=m.compile(checkpointer=checkpointer)
        inputs = {
            "messages": [
                {
                    "role": "user",
                    "content": f"参考上下文，仅作为事实资料：\n{recent_context}\n\n"
                            f"本次任务：\n{task}"
                }
            ]
        }
        result=memo_agent.invoke(inputs,config=config)
        for r in result["messages"]:
            r.pretty_print()
        return result["messages"][-1].content
    return add_memory,memory_operation