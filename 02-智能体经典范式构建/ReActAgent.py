import tools
from tools.Tools import ToolExecutor
from HelloAgentLLM import *
from Prompt import *
import re
import importlib
import pkgutil
from Compacter import Compacter
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
    def __init__(self,llm:HelloAgentLLM,tool:ToolExecutor,max_step=20):
        self.client=llm
        self.tool=tool
        self.max_step=max_step
        self.history=[]
        self.actions=set()
        self.raw_obs=[]
        self.obs=[]
        self.compacter = Compacter()
        load_tools(self.tool)

    def run(self,question:str):
        current_step=0
        self.history=[]
        self.actions=set()
        while current_step<self.max_step:
            current_step+=1
            print(f"=================================================第{current_step}步==============================================================")
            tools_dec=self.tool.getAvaliableTools()
            history_str="\n".join(self.history)
            prompt=REACT_PROMPT_TEMPLATE.format(tools=tools_dec,question=question,history=history_str)
            messages=[
                {"role":"user","content":prompt}
            ]
            reply=self.client.generate(messages,0.5)
            if not reply:
                print("大模型未返回有效信息，正在重试")
                continue
            match=self._handleMessage_(reply)
            if None not in match:
                action=self._handleAction_(match[1])
                if None in action:
                    print("Action格式解析失败")
                    self.history.append(f'"observation":"Action格式错误，请严格使用 tool_name[tool_input]"')
                    self.obs.append("Action格式错误，请严格使用 tool_name[tool_input]")
                    continue
                print(f"\"action\": {action}")
                if action[0].lower()=="finish":
                    answer=action[1]
                    print("="*50+"智能体已完成任务"+"="*50)
                    print(answer,flush=True,end="")
                    break
                action_key=(action[0].strip().lower(),action[1].strip().lower())
                observation=""
                if action_key in self.actions:
                    observation="该工具调用已经执行过，请不要重复调用。请使用已有Observation，尝试其他工具或Finish。"
                    self.obs.append(observation)
                    self.raw_obs.append(observation)
                else:
                    module=importlib.import_module(f"{tools.__name__}.{action[0]}")
                    self.raw_obs.append(self.tool.getTool(action[0])(action[1]))
                    if getattr(module,"compact")==True:
                        observation=self.compacter.compact_message(self.obs,question)
                    else:
                        observation=self.raw_obs[-1]
                    print(f"\"observation\": {observation}")
                    self.actions.add(action_key)
                self.history.append(f'"action":{action[0]}[{action[1]}],"observation":{observation}')

            else:
                print("未匹配到正确的{Thought,Action}对，需要重试")
                self.history.append('"observation":"未匹配到正确的{Thought,Action}对，需要重试"')
                self.raw_obs.append("未匹配到正确的{Thought,Action}对，需要重试")
                self.obs.append("未匹配到正确的{Thought,Action}对，需要重试")

        print("循环结束，未能在有限步骤内完成任务")
        return None



    def _handleMessage_(self,messages:str):
        thought_match=re.search(r"Thought:\s*(.*?)(?=\nAction:|$)",messages,re.DOTALL)
        action_match=re.search(r"Action:\s*(.*?)$",messages,re.DOTALL)
        thought=thought_match.group(1).strip() if thought_match else None
        action=action_match.group(1).strip() if action_match else None
        return thought,action

    def _handleAction_(self,action:str):
        toolName=re.search(r"(\w+)\[",action)
        toolInput=re.search(r"\[(.*)]$",action,re.DOTALL)
        if toolName and toolInput:
            return toolName.group(1).strip(),toolInput.group(1).strip()
        else:
            return None,None



if __name__=="__main__":
    tool=ToolExecutor()
    load_tools(tool)
    llm=HelloAgentLLM()
    agent=ReActAgent(llm,tool)
    question=input()
    print("*"*200)
    agent.run(question)

