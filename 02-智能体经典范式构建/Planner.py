import os
from HelloAgentLLM import*
from Prompt import *
import tools
from Reflection import ReflectionAgent
from tools.Tools import *
import re
from openai import OpenAI
from typing import List, Dict, Any
from ReActAgent import ReActAgent,load_tools
from dotenv import load_dotenv
import time
from Memory import *
from ast import literal_eval
load_dotenv()

class Planner:
    def __init__(self,llm:HelloAgentLLM):
        self.planner = llm
        self.plan=[]
        self.tool=ToolExecutor()
        self.react_solver=ReActAgent(llm,self.tool,8)
        self.common_solver=HelloAgentLLM()
        self.obs=[]
        self.reflection=ReflectionAgent(llm)

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

    def solve(self,state:AgentState,question:str,plan:List[Dict[str,Any]]):
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
        history_result = []
        i = 1

        for subplan in plan:
            flag = False

            for j in range(1, 4):
                print(
                    f"{i}. {subplan.get('task')} "
                    f"[第 {j}/3 次尝试]"
                )

                executor_context = EXECUTOR_PROMPT_TEMPLATE.format(
                    question=question,
                    plan=plan,
                    history=history_result,
                    current_step=subplan.get("task"),
                    history_obs=self.obs
                )

                # ==================== 需要工具 ====================
                if subplan.get("need_tool"):
                    try:
                        reply, raw_obs, obs = self.react_solver.run(
                            subplan.get("task"),
                            executor_context,
                            temperature=0.3
                            ,state=state
                        )

                        if obs and obs[-1] == "success":
                            self.obs.append(obs[:-1])

                            history_result.append({
                                "task": subplan.get("task"),
                                "observation": obs[:-1],
                                "tackle": reply
                            })

                            flag = True
                            break

                        else:
                            print(f"当前任务第 {j} 次执行失败")

                    except Exception as e:
                        print(f"当前任务第 {j} 次发生异常：{e}")

                # ==================== 不需要工具 ====================
                else:
                    try:
                        reply = self.common_solver.generate(
                            [{
                                "role": "user",
                                "content": executor_context
                            }],
                            0,
                            True
                        )

                        if reply:
                            history_result.append({
                                "task": subplan.get("task"),
                                "observation": None,
                                "tackle": reply
                            })

                            flag = True
                            break

                    except Exception as e:
                        print(f"当前任务第 {j} 次发生异常：{e}")

            # 三次尝试完成以后
            if flag:
                print(f"{i}. {subplan.get('task')} [x]")
            else:
                history_result.append({
                    "task": subplan.get("task"),
                    "observation": None,
                    "tackle": "该步骤连续执行3次均失败"
                })
                print(f"{i}. {subplan.get('task')} [失败]")

            i += 1

        # ==================== 所有子任务执行完才总结 ====================

        summarizer = HelloAgentLLM()

        summarizer_prompt = SUMMARY_PROMPT_TEMPLATE.format(
            question=question,
            observation=self.obs,
            history=history_result
        )

        answer = summarizer.generate(
            [{"role": "user", "content": summarizer_prompt}],
            0.1
        )

        return answer

if __name__ == "__main__":
    start=time.perf_counter()
    llm = HelloAgentLLM()
    planer = Planner(llm)
    question="""这是一个异常处理测试，请严格执行：

1. 不要调用 search。
2. 直接调用 read_urls 工具。
3. tool_input 只传入：
   {
       "urls": ["https://example.invalid/reflection-trigger"]
   }
4. 故意不要传入 question 字段，不要自动补全。
5. 根据工具返回的错误信息，分析这次工具调用失败的原因，并给出下一步建议。
6. 最后调用 Finish。"""
    print("+"*200)
    state = AgentState()
    plan=planer.planer(question)
    state.plan = plan
    state.task_id = "001"

    print("----------plan------------")
    print(plan)
    print(planer.solve(state,question,plan))

    endtime=time.perf_counter()
    print(f"共计用时：{endtime-start:.2f}s")
