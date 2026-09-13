import os
from HelloAgentLLM import*
from Prompt import *
import tools
from tools.Tools import *
import re
from openai import OpenAI
from typing import List, Dict, Any
from ReActAgent import ReActAgent,load_tools
from dotenv import load_dotenv
import time
from ast import literal_eval
load_dotenv()

class Planner:
    def __init__(self,llm:HelloAgentLLM):
        self.planner = llm
        self.plan=[]
        self.tool=ToolExecutor()
        self.react_solver=ReActAgent(llm,self.tool,20)
        self.common_solver=HelloAgentLLM()
        self.obs=[]

    def planer(self,question:str)->List:
        self.plan=[]
        prompt = PLANNER_PROMPT_TEMPLATE.format(question=question,tool=self.tool)
        messages=[{"role":"user","content":prompt}]
        response=self.planner.generate(messages,0.1) or ""
        if not response:
            print("生成计划失败，请重试")
            return ["生成计划失败，请重试"]
        match = re.search(r"```python\s*(.*?)\s*```", response, re.DOTALL)

        if not match:
            print("计划生成失败：没有找到 Python 代码块")
            print("模型原始输出：")
            print(response)
            return []

        try:
            self.plan = literal_eval(match.group(1))
        except (ValueError, SyntaxError) as e:
            print(f"计划解析失败：{e}")
            print("原始计划内容：")
            print(match.group(1))
            return []
        return self.plan

    def solve(self,question:str,plan:List[Dict[str,Any]]):
        self.obs=[]
        if not isinstance(plan,list):
            print("plan格式错误，必须是List[Dict[any,any]]的格式，外层必须是列表")
            return "plan格式错误，必须是List[Dict[any,any]]的格式，外层必须是列表"
        else:
            if not all(isinstance(subplan,dict)for subplan in plan):
                print("plan格式错误，必须是List[Dict[any,any]]的格式，内层必须是字典")
                return "plan格式错误，必须是List[Dict[any,any]]的格式，内层必须是字典"

        if len(plan)==0 and question:
            print("任务列表为空，退化到ReActAgent执行")
            reply,raw_obs,obs=self.react_solver.run(question)
            return reply
        if not question:
            print("错误！请输入问题")
            return "问题为空，无法执行"
        if not all(plan[j].get("task") and "need_tool" in plan[j] for j in range(len(plan))):
            print("plan格式错误，注意每一个子任务都需要带有'task'和'need_tool'字段")
            return "plan格式错误，注意每一个子任务都需要带有'task'和'need_tool'字段"
        history_result=[]
        i=1
        for subplan in plan:
            print(f"{i}. {subplan.get('task')} []")
            reply=""
            if subplan.get("need_tool"):
                reply,raw_obs,obs=self.react_solver.run(subplan.get("task"),temperature=0.3)
                self.obs.append(obs)
                history_result.append({"task":subplan.get('task'),"observation":obs,"tackle":reply})
            else:
                reply=self.common_solver.generate([{"role": "user",
                "content":EXECUTOR_PROMPT_TEMPLATE.format(question=question,plan=plan,history=history_result,current_step=subplan.get('task'),history_obs=self.obs)}]
                ,0,True)
                history_result.append({"task":subplan.get('task'),"observation":None,"tackle":reply})

            print(f"{i}. {subplan.get('task')} [x]")
            i=i+1

        summarizer=HelloAgentLLM()
        summarizer_prompt=SUMMARY_PROMPT_TEMPLATE.format(question=question,observation=self.obs,history=history_result)
        answer = summarizer.generate([{"role":"user","content":summarizer_prompt}],0.1)
        return answer


if __name__ == "__main__":
    start=time.perf_counter()
    llm = HelloAgentLLM()
    planer = Planner(llm)
    question="请比较 GPT-5.6、Claude Opus 4.8 和 Gemini 3.8 flash 在 2026 年 9 月的最新能力、API 价格、上下文窗口和编程表现，并结合“学生个人开发者，主要用于 Agent 开发和代码调试，每月预算 100 元人民币以内”的条件，给出最适合我的选择。要求优先查官方资料；如果官方没有编程能力对比，再查可信的第三方评测。"
    print("+"*200)
    plan=planer.planer(question)
    print("----------plan------------")
    print(plan)
    print(planer.solve(question,plan))

    endtime=time.perf_counter()
    print(f"共计用时：{endtime-start:.2f}s")
