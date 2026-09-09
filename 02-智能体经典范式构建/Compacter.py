import os

from HelloAgentLLM import HelloAgentLLM
from dotenv import load_dotenv
load_dotenv()

class Compacter:
    def __init__(self,llm:HelloAgentLLM=None):
        self.compacter=HelloAgentLLM(model=os.getenv("COMPACTER_MODEL"))


    def compact_message(self,observation:str,question:str):
        prompt = f"""
        你负责压缩Agent工具返回结果。

        用户原始问题：
        {question}

        以下是工具Observation：

        <observation>
        {observation}
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
        10. 输出控制在 1000 字以内。
        11.只输出压缩后的 Observation，不要回答用户问题。
        12. 控制在2000字以内。
        """
        message=[{"role":"user","content":prompt}]
        text=self.compacter.generate(message,0.1,False)
        while len(text)>2100:
            message.append({"role":"assistant","content":text})
            message.append({"role":"user","content":f"字数超出了限制，你需要再缩略至少{len(text)-2100}字"})
            text=self.compacter.generate(message,0.1,False)

        return text