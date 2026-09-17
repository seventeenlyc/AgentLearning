import os
from typing import List
from HelloAgentLLM import HelloAgentLLM
from dotenv import load_dotenv
load_dotenv()

class Compacter:
    def __init__(self,llm:HelloAgentLLM=None):
        self.compacter=HelloAgentLLM(model=os.getenv("COMPACTER_MODEL"))


    def compact_message(self,observation:List,question:str)->str:
        print("开始压缩信息")
        prompt = """
        你负责压缩Agent工具返回结果。

        用户原始问题：
        {question}

        以下是工具Observation：

        <observation>
        {info}
        </observation>

        请仅保留与用户问题有关的信息。

        要求：
        严格遵守：
        1. 只能使用 Observation 中明确出现的信息。
        2. 严禁利用你自己的知识补充信息。
        3. 严禁推测、猜测或生成 Observation 中没有出现的数字、日期、名称和结论。
        4. 原文没有说明的内容，不要写。
        5. 保留与用户问题直接相关的重要事实。
        6. 保留关键数字、时间、人物、名称、网址。
        7. 删除导航栏、广告、推荐文章、页脚等无关内容。
        8. 对不确定的信息明确标记为“原文未明确说明”。
        9. 尽量保留信息来源的措辞和事实关系。
        10.只输出压缩后的 Observation，不要回答用户问题。
        11. 控制在2500字以内。
        12. 在末尾加上涉及的网址。
        """
        info:str=""
        if isinstance(observation,str):
            info=observation
        else:
            for i, item in enumerate(observation):
                info += f"{i}. {item}\n"
        message=[{"role":"user","content":prompt.format(question=question,info=info)}]
        text=self.compacter.generate(message,0.1,False)
        retry=0
        while len(text)>2500 and retry<3:
            retry+=1
            message.append({"role":"assistant","content":text})
            message.append({"role":"user","content":f"字数超出了限制，你需要再缩略至少{len(text)-2500}字，但"})
            text=self.compacter.generate(message,0.1,False)
        print("压缩结束")
        return text