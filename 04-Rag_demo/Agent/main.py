import re
from ast import literal_eval
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, MessagesState, START, END
import os,sqlite3,requests
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from datetime import date, datetime, timezone, timedelta
from typing import Literal
from uuid import uuid4
from Agent.memory_service import MemorySemantic,create_memory
from Agent.memory_agent import memory_agent
from Agent.tools import calc,get_weather,build_memory_tools,get_time,google_search,gemini_google_search,read_url
from retriever import Retriever

load_dotenv()

if __name__ == "__main__":
    conn = sqlite3.connect(
        "session.db",
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row

    conn.execute("""
                 CREATE TABLE IF NOT EXISTS session
                 (
                     thread_id    TEXT PRIMARY KEY NOT NULL,
                     title        TEXT             NOT NULL DEFAULT '新会话',
                     created_at   TEXT             NOT NULL DEFAULT CURRENT_TIMESTAMP,
                     updated_at   TEXT             NOT NULL DEFAULT CURRENT_TIMESTAMP,
                     title_source TEXT             NOT NULL DEFAULT 'pending'
                 )""")
    checkpointer = SqliteSaver(conn)
    thread_id = str(uuid4())
    config={
        # 父、子 Agent 共用同一检索器，串行执行工具避免修改与删除相互覆盖。
        "configurable": {
            "thread_id": thread_id,
        }
    }

    main_model = ChatOpenAI(model=os.getenv("LLM_MODEL"),
                       api_key=os.getenv("LLM_API_KEY"),
                       base_url=os.getenv("BASE_URL"), timeout=30,max_retries=0)

    small_model=ChatOpenAI(model=os.getenv("SMALL_MODEL"),
                           api_key=os.getenv("SMALL_API_KEY"),
                           base_url=os.getenv("SMALL_BASE_URL"))
    retriever = Retriever(path="../data/memories.json")
    add_memory,memory_operation=memory_agent(ret=retriever,model=main_model,config=config)
    tools=[calc,get_weather,memory_operation,get_time,google_search,gemini_google_search,read_url]

    model_with_tools=main_model.bind_tools(tools)


    def call_model(state: MessagesState):
        response = model_with_tools.invoke(state["messages"])
        return {
            "messages": [response],
        }

    g=StateGraph(MessagesState)
    g.add_node("model",call_model)
    g.add_node("tools",ToolNode(tools))

    g.add_edge(START,"model")
    g.add_conditional_edges("model",tools_condition)
    g.add_edge("tools","model")


    app=g.compile(checkpointer=checkpointer)
    inputs = lambda question: {
        "messages": [
            {
                "role": "user",
                "content": question
            }
        ]
    }


    try:
        while True:
            print('-'*100)
            question=input("输入问题：\n")

            if question.lower()=="exit" or question.lower()=="/exit" or question.lower()=="q":
                break
            elif question.lower()=="/session":
                rows=conn.execute("""
                SELECT thread_id,title,datetime(updated_at, '+8 hours') AS updated_at FROM session
                ORDER BY updated_at DESC""").fetchall()
                print("thread_id",'\t',"title",'\t',"updated_at")
                for r in rows:
                    print(r["thread_id"],'\t',r["title"],'\t',r["updated_at"])
                continue
            elif question.startswith("/resume"):
                target_id=question[len("/resume"):].strip()
                find_target=conn.execute("""SELECT thread_id FROM session WHERE thread_id=?""",(target_id,)).fetchone()
                if not find_target:
                    print(f"无法找到会话{target_id}，请运行/session查看会话id")
                    continue
                thread_id=target_id
                config["configurable"]["thread_id"]=thread_id
                print(f"已切换到会话：{thread_id}")
                # 读取该会话保存的最新状态
                snapshot = app.get_state(config)
                messages = snapshot.values.get("messages", [])
                # 打印原来的聊天记录
                for message in messages:
                    message.pretty_print()
                continue
            elif question.lower()=="/new":
                thread_id=str(uuid4())
                config["configurable"]["thread_id"]=thread_id
                print("新会话已创建，会话id{thread_id}".format(thread_id=thread_id))
                continue
            elif question.lower().startswith("/rename"):
                cur_title=conn.execute("SELECT title FROM session WHERE thread_id=?",
                                       (thread_id,)).fetchone()
                new_title_input=input(f"当前标题为:{cur_title["title"]}\n请输入新标题，输入回车表示取消重命名\n")
                new_title=new_title_input.strip()
                if not new_title:
                    continue
                conn.execute("""UPDATE session SET title=?,title_source=?,updated_at=CURRENT_TIMESTAMP
                             WHERE thread_id=?""",(new_title,"manual",thread_id,))
                print(f"标题已更新，新标题为：{new_title}")
                continue

            elif question.lower().startswith("/clear"):
                target_id=question[len("/clear"):].strip() or thread_id
                find_id=conn.execute("""SELECT thread_id FROM session WHERE thread_id=?""",(target_id,)).fetchone()
                if not find_id:
                    print("找不到会话")
                    continue
                if target_id==thread_id:
                    thread_id=str(uuid4())
                    config["configurable"]["thread_id"]=thread_id
                conn.execute("""DELETE FROM session WHERE thread_id=?""",(target_id,))
                checkpointer.delete_thread(target_id)
                print(f"会话{target_id}已删除")
                continue

            print("正在调用主模型……", flush=True)
            result=app.invoke(inputs(question),config=config)

            result["messages"][-1].pretty_print()
            # 创建会话记录，并更新最近聊天时间
            with conn:
                find_thread = conn.execute(
                    "SELECT thread_id FROM session WHERE thread_id = ?",
                    (thread_id,),
                ).fetchone()

                if find_thread is None:
                    conn.execute(
                        "INSERT INTO session (thread_id) VALUES (?)",
                        (thread_id,),
                    )

                conn.execute(
                    "UPDATE session SET updated_at = CURRENT_TIMESTAMP WHERE thread_id = ?",
                    (thread_id,),
                )

            # 查询标题来源
            row = conn.execute(
                "SELECT title_source FROM session WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()

            if row is not None and row[0] == "pending":
                response = small_model.invoke([
                    {
                        "role": "system",
                        "content": "根据用户问题生成一个简短中文标题，只输出标题。",
                    },
                    {
                        "role": "user",
                        "content": question,
                    },
                ])
                text=response.content.strip()
                title=re.search(r"<thought>.*\s*</thought>(.*?)$",text,re.DOTALL)
                with conn:
                    conn.execute("""
                                 UPDATE session
                                 SET title_source = 'auto',
                                     title        = ?
                                 WHERE thread_id = ?
                                   AND title_source = 'pending'
                                 """, (title.group(1), thread_id))


    finally:
        conn.close()