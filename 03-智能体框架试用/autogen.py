import asyncio

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from dotenv import load_dotenv
import os
load_dotenv()
def create_client(model=None,key=None,url=None,family=None):
    return OpenAIChatCompletionClient(
        model=model or os.getenv('ASTRAFLOW_MODEL'),
        api_key=key or os.getenv('ASTRAFLOW_API_KEY'),
        base_url=url or os.getenv('ASTRAFLOW_BASE_URL'),
        model_info={
            "function_calling": True,
            "vision": False,
            "json_output": True,
            "family": family or "deepseek",
            "structured_output": True
        }
    )

def create_playerA(model_client):
    system_message = """
    你是deepseek，1号玩家。

    你知道的关键词是**西红柿炒蛋**，你有可能是平民也有可能是卧底。

    游戏规则：
    现在正在进行一场游戏，场上有五位选手，对应四名平民和一名卧底。平民知道正确的关键词，而卧底知道的关键词是错误的。
    0. 四名选手分别是：Deepseek、Mimo、glm、qwen、拾叁。
    1. 四名选手依次发言分享信息，直到四名选手都发言三次以后，进入投票环节，四名选手指认卧底，被半数以上玩家投票的玩家将被驱逐。
    2. 如果卧底被驱逐，则平民获胜；如果有平民被驱逐，则卧底获胜。
    3. 如果在一次投票中，没有选出驱逐的玩家，则四位玩家再发言一次，重新投票。重复此操作直到有玩家被驱逐。
    
    你的规则：
    - 不可以把关键词以各种方式说出，只能描述它的用途、拥有情况等信息，一次性不要描述的太多太全面，只说一个点。
    - 你可以通过发言试探其他玩家
    - 找出卧底/隐藏自己的卧底身份
    - 描述关键词时，遵循结构：1号玩家{{name}}的第？轮发言：……
    - 投票时，可以分析，但在最后一句话请输出：所以我投票给{{who}}
    - 如果你发现你是卧底，可以猜测并描述平民的关键词，而不是自己的关键词。
    """
    return AssistantAgent(
        name="Deepseek",
        model_client=model_client,
        system_message=system_message,
        model_client_stream=True,
    )

def create_playerB(model_client):
    system_message = """
    你是xiaomi，2号玩家。

    你知道的关键词是**酸辣土豆丝**，你有可能是平民也有可能是卧底。

    游戏规则：
    现在正在进行一场游戏，场上有五位选手，对应四名平民和一名卧底。平民知道正确的关键词，而卧底知道的关键词是错误的。
    0. 四名选手分别是：Deepseek、Mimo、glm、qwen、拾叁。
    1. 四名选手依次发言分享信息，直到四名选手都发言三次以后，进入投票环节，四名选手指认卧底，被半数以上玩家投票的玩家将被驱逐。
    2. 如果卧底被驱逐，则平民获胜；如果有平民被驱逐，则卧底获胜。
    3. 如果在一次投票中，没有选出驱逐的玩家，则四位玩家再发言一次，重新投票。重复此操作直到有玩家被驱逐。
    
    你的规则：
    - 不可以把关键词以各种方式说出，只能描述它的用途、拥有情况等信息，一次性不要描述的太多太全面，只说一个点。
    - 你可以通过发言试探其他玩家
    - 找出卧底/隐藏自己的卧底身份
    - 描述关键词时，遵循结构：2号玩家{{name}}的第？轮发言：……
    - 投票时，可以分析，但在最后一句话请输出：所以我投票给{{who}}
    - 如果你发现你是卧底，可以猜测并描述平民的关键词，而不是自己的关键词。
    请你用中文思考
    """
    return AssistantAgent(
        name="xiaomi",
        model_client=model_client,
        system_message=system_message,
        model_client_stream=True,
    )

def create_playerC(model_client):
    system_message = """
    你是glm，3号玩家。

    你知道的关键词是**西红柿炒蛋**，你有可能是平民也有可能是卧底。

    游戏规则：
    现在正在进行一场游戏，场上有五位选手，对应四名平民和一名卧底。平民知道正确的关键词，而卧底知道的关键词是错误的。
    0. 四名选手分别是：Deepseek、Mimo、glm、qwen、拾叁。
    1. 四名选手依次发言分享信息，直到四名选手都发言三次以后，进入投票环节，四名选手指认卧底，被半数以上玩家投票的玩家将被驱逐。
    2. 如果卧底被驱逐，则平民获胜；如果有平民被驱逐，则卧底获胜。
    3. 如果在一次投票中，没有选出驱逐的玩家，则四位玩家再发言一次，重新投票。重复此操作直到有玩家被驱逐。
    
    你的规则：
    - 不可以把关键词以各种方式说出，只能描述它的用途、拥有情况等信息，一次性不要描述的太多太全面，只说一个点。
    - 你可以通过发言试探其他玩家
    - 找出卧底/隐藏自己的卧底身份
    - 描述关键词时，遵循结构：3号玩家{{name}}的第？轮发言：……
    - 投票时，可以分析，但在最后一句话请输出：所以我投票给{{who}}
    - 如果你发现你是卧底，可以猜测并描述平民的关键词，而不是自己的关键词。
    
    请你用中文思考
    """
    return AssistantAgent(
        name="glm",
        model_client=model_client,
        system_message=system_message,
        model_client_stream=True,
    )


def create_playerD(model_client):
    system_message = """
    你是qwen，4号玩家。

    你知道的关键词是**西红柿炒蛋**，你有可能是平民也有可能是卧底。

    游戏规则：
    现在正在进行一场游戏，场上有四位选手，对应三名平民和一名卧底。平民知道正确的关键词，而卧底知道的关键词是错误的。
    0. 四名选手分别是：Deepseek、Mimo、glm、qwen、拾叁。
    1. 四名选手依次发言分享信息，直到四名选手都发言三次以后，进入投票环节，四名选手指认卧底，被半数以上玩家投票的玩家将被驱逐。
    2. 如果卧底被驱逐，则平民获胜；如果有平民被驱逐，则卧底获胜。
    3. 如果在一次投票中，没有选出驱逐的玩家，则四位玩家再发言一次，重新投票。重复此操作直到有玩家被驱逐。

    你的规则：
    - 不可以把关键词以各种方式说出，只能描述它的用途、拥有情况等信息，一次性不要描述的太多太全面，只说一个点。
    - 你可以通过发言试探其他玩家
    - 找出卧底/隐藏自己的卧底身份
    - 描述关键词时，遵循结构：4号玩家{{name}}的第？轮发言：……
    - 投票时，可以分析，但在最后一句话请输出：所以我投票给{{who}}
    - 如果你发现你是卧底，可以猜测并描述平民的关键词，而不是自己的关键词。

    """
    return AssistantAgent(
        name="qwen",
        model_client=model_client,
        system_message=system_message,
        model_client_stream=True,
    )

def create_player():
    return UserProxyAgent(name="shisan")
def create_judger(model_client):
    system_message = """
    现在正在进行一场游戏，场上有五位选手，对应四名平民和一名卧底。平民知道正确的关键词，而卧底知道的关键词是错误的。
    关键词是**西红柿炒蛋**，卧底是Xiaomi。
    现在你将作为这场游戏的裁判，游戏规则：
    1. 四名选手依次发言分享信息，直到四名选手都发言三次以后，进入投票环节，四名选手指认卧底，被半数以上玩家投票的玩家将被驱逐。
    2. 如果卧底被驱逐，则平民获胜；如果有平民被驱逐，则卧底获胜。
    3. 如果在一次投票中，没有选出驱逐的玩家，则四位玩家再发言一次，重新投票。重复此操作直到有玩家被驱逐。
    
    作为裁判，你**不能**泄露四名选手的身份和信息，你只能进行：
    1. 如果四名选手还未完成所有发言，提示他们继续发言。
    2. 如果四名选手均发言结束，提示他们开始投票。
    3. 如果投票后没有玩家被驱逐，公布每一个玩家的票数，提示他们再次分享信息。
    4. 如果投票后有玩家被驱逐，公布每一个玩家的票数，宣布游戏结束，宣判游戏获胜方。
    """
    return AssistantAgent(
        name="judger",
        model_client=model_client,
        system_message=system_message,
        model_client_stream=True,
    )


playerA=create_playerA(create_client())
playerB=create_playerB(create_client(model="mimo-v2.6-flash",family="mimo"))
playerC=create_playerC(create_client(model="glm-5.3-flash",family="glm"))
playerD=create_playerD(create_client(model="qwen3.7-plus",family="qwen"))
judger=create_judger(create_client())
player=create_player()
team_chat=RoundRobinGroupChat(
participants=[playerA,playerB,playerC,playerD,player,judger],
    max_turns=30
)
async def run_software_development_team():
    # ... 初始化客户端和智能体 ...

    # 定义任务描述
    task = """游戏开始，请根据自己的私有信息进行讨论"""

    # 异步执行团队协作，并流式输出对话过程
    result = await Console(team_chat.run_stream(task=task))
    return result


# 主程序入口
if __name__ == "__main__":
    result = asyncio.run(run_software_development_team())
