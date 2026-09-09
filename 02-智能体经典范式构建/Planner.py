import os
from HelloAgentLLM import*
from Prompt import *
import tools
from tools.Tools import *
import re
from openai import OpenAI
from typing import List
from ReActAgent import ReActAgent,load_tools
from dotenv import load_dotenv
load_dotenv()

class Planner:
    def __init__(self,llm:HelloAgentLLM):
        self.planner = llm
        self.planer=[]
        self.solver=llm

    def plan(self,question:str)->List:
        self.plan=[]
        prompt = PLANNER_PROMPT_TEMPLATE.format(question=question)
        messages=[{"role":"user","content":prompt}]
        response=self.planner.generate(messages,0.1) or ""
        matches=re.findall(r"\"(.*?)\"",response,)
        for match in matches:
            self.plan.append(match)
        print(f"计划已生成，共{len(self.plan)}步")
        for i in range(len(self.plan)):
            print(f"[]{i+1}. {self.plan[i]}")
        return self.plan or []


    def solve(self,question:str,plan:List):
        if len(plan)==0:
            print("错误！任务列表为空！")
            return

        history=[]
        i=1
        for current_step in plan:
            print(f"{i}. {current_step}")
            prompt = EXECUTOR_PROMPT_TEMPLATE.format(question=question,plan=plan,history=history,current_step=current_step)
            messages=[{"role":"user","content":prompt}]
            response=self.planner.generate(messages,0.1,output=False) or ""
            if not response:
                print("无输出，请重试")
                history.append(f"{current_step}:无输出")
            history.append(f"{current_step}:{response}")
            print()
            print(f"[x]{i}.{current_step}")
            print(response,end="",flush=True)
            print()
            i+=1


if __name__ == "__main__":
    llm = HelloAgentLLM()
    planner = Planner(llm)
    question="一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？"
    print("+"*50)
    plan=planner.plan(question)
    planner.solve(question,plan)
