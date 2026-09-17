import tools
from tools.Tools import ToolExecutor
from HelloAgentLLM import *
from Prompt import *
import re
import importlib
import pkgutil
from Compacter import Compacter
import ast
import json
from typing import Dict,List,Any
def load_tools(tool:ToolExecutor):
    for info in pkgutil.iter_modules(tools.__path__):
        module_name=info.name
        if module_name=="Tools" or module_name=="__init__":
            continue
        module=importlib.import_module(f"{tools.__name__}.{module_name}")
        name=getattr(module,"name",None)
        description=getattr(module,"description",None)
        func=getattr(module,"func",None)
        if name and description and func:
            tool.registerTool(name,description,func)
        else:
            print(f"跳过非法工具模块{module_name}")




class ReActAgent:
    def __init__(self,llm:HelloAgentLLM,tool:ToolExecutor,max_step=8):
        self.client=llm
        self.tool=tool
        self.max_step=max_step
        self.history=[]
        self.actions=set()
        self.raw_obs=[]     #记录工具的原始调用结果
        self.obs=[]     #raw_obs经过处理以后得到的高质量的Observation
        self.compacter = Compacter()
        load_tools(self.tool)

    def run(self,question:str,context:str="",sys_prompt:str=None,temperature:float=0.5)->(str,List,List):
        """
        ReActAgent的运行入口
        :param question:
        :return: 返回模型根据提问的最终结果、原始的工具调用结果、处理后的高质量Observation
        """

        self.history=[]
        self.obs=[]
        self.raw_obs=[]
        self.actions=set()
        for r in range(1,4):
            current_step = 0
            while current_step<self.max_step:
                current_step+=1
                remain_step=self.max_step-current_step
                print(f"=================================================第{current_step}轮==============================================================")
                tools_dec=self.tool.getAvaliableTools()
                history_str="\n".join(self.history)
                step_warning=""
                if current_step==self.max_step-1:
                    step_warning="你只剩最后一次机会，优先收敛信息，准备在下一轮输出答案。"
                elif current_step==self.max_step:
                    step_warning="当前为最后一轮，禁止调用工具，只允许 Finish。"
                prompt=REACT_PROMPT_TEMPLATE.format(
                    tools=tools_dec,question=question,history=history_str,
                    current_step=current_step,max_step=self.max_step,remaining_steps=remain_step,
                    step_warning=step_warning,
                    context=context
                )
                messages=[]
                if sys_prompt:
                    messages.append({"role":"system","content":sys_prompt})
                messages.append({"role":"user","content":prompt})

                reply=self.client.generate(messages,temperature=temperature)
                if not reply:
                    print("大模型未返回有效信息，正在重试")
                    continue
                match=self._handleMessage_(reply)
                if match.get("Action") is None:
                    print("未找到Action字段，需要重试")
                    self.history.append(
                        '"observation":"没有检测到Action字段，请严格按照Thought和Action格式输出"'
                    )
                    continue
                tool_name,tool_input=self._handleAction_(match)
                if not tool_name:
                    print("Action格式解析失败")
                    self.history.append(f'"observation":"Action格式错误，请严格遵循Action格式"')
                    continue
                print(f"\naction: {tool_name}[{tool_input}]")
                if tool_name.lower()=="finish":
                    if tool_input.get('status')=='success' and tool_input.get('answer'):
                        print('*'*200)
                        print(tool_input.get('answer'),flush=True,end='')
                        self.obs.append("success")
                        return tool_input.get('answer'),self.raw_obs,self.obs
                    elif tool_input.get('status')=='failed' and tool_input.get('reason'):
                        print("任务失败，原因" + tool_input.get('reason'))
                        self.history.append("任务失败，原因"+tool_input.get('reason'))
                        self.raw_obs.append(tool_input.get('reason'))
                        break
                    else:
                        print("finish格式错误，缺少参数")
                        self.history.append("finish格式错误，缺少参数")
                        continue

                if current_step==self.max_step and tool_name.lower()!="finish":
                    final_prompt = f"""
                    根据以下已有执行历史，输出当前最佳答案。
                    不得调用工具，不得补充历史中不存在的事实。
    
                    {self.history}
                    """
                    answer=self.client.generate([{"role":"user","content":final_prompt}],temperature=temperature)
                    return answer,self.raw_obs,self.obs
                action_key = (
                    tool_name.lower(),
                    json.dumps(tool_input, sort_keys=True, ensure_ascii=False)
                )
                observation=""
                if action_key in self.actions:
                    observation="该工具调用已经执行过，请不要重复调用。请使用已有Observation，尝试其他工具或Finish。"
                else:
                    module=importlib.import_module(f"{tools.__name__}.{tool_name}")
                    self.raw_obs.append(self.tool.getTool(tool_name)(tool_input))
                    if getattr(module,"compact")==True:
                        observation=self.compacter.compact_message(self.raw_obs[-1],question)
                    else:
                        observation=self.raw_obs[-1]
                    print(f"\"observation\": {observation}")
                    self.actions.add(action_key)
                self.history.append(f'"action":{tool_name}[{tool_input}],"observation":{observation}')
                self.obs.append(observation)



        print("循环结束，未能在有限步骤内完成任务")
        return "未能在有限步骤内完成任务",self.raw_obs,self.obs



    def _handleMessage_(self,messages:str):
        """
        返回摘出来的Thought-Action对
        :param messages: 大模型原始的输出信息
        :return: Thought-Action对
        """
        match=re.search(r"```python\s(.*?)\s```",messages,re.DOTALL)
        if not match:
            print("未找到Python代码块")
            return {}
        message=match.group(1)
        try:
            reply=ast.literal_eval(message)
        except (ValueError, SyntaxError) as e:
            print(f"解析失败：{e}")
            return {}
        if not isinstance(reply, dict):
            print("计划不是Dict")
            return {}
        action = reply.get("Action")
        if not action:
            print("缺少Action字段")
            return {}
        if not isinstance(action, dict):
            print("Action不是Dict")
            return {}
        return reply

    def _handleAction_(self, reply:Dict) -> tuple[str | None, Dict | None]:
        action = reply.get("Action")
        if not isinstance(action, dict):
            return None, None

        tool_name = action.get("tool_name")
        tool_input = action.get("tool_input")

        if not tool_name or tool_input is None:
            return None, None

        return tool_name, tool_input


if __name__=="__main__":
    tool=ToolExecutor()
    llm=HelloAgentLLM()
    agent=ReActAgent(llm,tool,5)
    question=input()
    print("*"*200)
    agent.run(question)

