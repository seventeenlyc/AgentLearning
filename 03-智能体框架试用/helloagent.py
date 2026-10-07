from hello_agents import *
from dotenv import load_dotenv
import os
load_dotenv()


llm=HelloAgentsLLM(model=os.getenv('LLM_MODEL'),api_key=os.getenv('LLM_API_KEY'),base_url=os.getenv('BASE_URL'))

agent=SimpleAgent(name="AI Assistant",llm=llm,system_prompt="你是一个有用的AI助手")
print(agent.run("你好"))