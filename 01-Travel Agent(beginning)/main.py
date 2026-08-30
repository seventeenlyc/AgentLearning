import os
import re
import tool
from llm import LLMClient

LLM_API_KEY="ollama"
TAVILY_API_KEY=os.environ.get("TAVILY_API_KEY")
LLM_MODEL="hf.co/HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive:Q4_K_M"
BASE_URL="http://127.0.0.1:11434/v1"

llm=LLMClient(LLM_MODEL,BASE_URL,LLM_API_KEY)
SYSTEM_PROMPT="""
你是一个智能旅行助手。你的任务是分析用户的请求，并使用可用工具一步步地解决问题。

# 可用工具:
- `get_weather(city: str)`: 查询指定城市的实时天气。
- `get_attraction(city: str, weather: str)`: 根据城市和天气搜索推荐的旅游景点。

# 输出格式要求:
你的每次回复必须严格遵循以下格式，包含一对Thought和Action，不可输出其他内容：

Thought: [你的思考过程和下一步计划]
Action: [你要执行的具体行动]

Action的格式必须是以下之一：
1. 调用工具：function_name(arg_name="arg_value")
2. 结束任务：Finish[最终答案]

# 重要提示:
- 每次只输出一对Thought-Action
- Action必须在同一行，不要换行
- 当收集到足够信息可以回答用户问题时，必须使用 Action: Finish[最终答案] 格式结束

请开始吧！
"""


user_prompt=input("输入你的问题\n")
print('='*500)
prompt_history=[f"用户请求:{user_prompt}"]


for i in range(10):
    print(f"第{i+1}次轮询...")

    full_prompt='\n'.join(prompt_history)
    llm_output=llm.generate(full_prompt,SYSTEM_PROMPT)

    print(llm_output)

    action=re.search(r"Action: (.*)", llm_output, re.DOTALL)
    if not action:
        tips="警告！未能解析到 Action 字段。请确保你的回复严格遵循 'Thought: ... Action: ...' 的格式。"
        print(tips)
        print('*'*500)
        prompt_history.append(f"Observation: {tips}")
        continue

    action_str=action.group(1).strip()

    if action_str.startswith("Finish"):
        final_answer = re.match(r"Finish\[(.*)\]", action_str).group(1)
        print(f"任务完成，最终答案: {final_answer}")
        break

    tool_name = re.search(r"(\w+)\(", action_str).group(1)
    args_str = re.search(r"\((.*)\)", action_str).group(1)
    kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))

    if tool_name in tool.available_tools:
        observation = tool.available_tools[tool_name](**kwargs)
    else:
        observation = f"错误:未定义的工具 '{tool_name}'"
    observation_str = f"Observation: {observation}"
    print(f"{observation_str}\n" + "="*500)
    prompt_history.append(observation_str)
