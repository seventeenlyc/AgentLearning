import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

class HelloAgentLLM:
    def __init__(self,model:str=None,apikey:str=None,base_url_:str=None,timeout:int=None):
        """
        初始化客户端
        """

        self.model=model or os.getenv("LLM_MODEL")
        self.apikey=apikey or os.getenv("LLM_API_KEY")
        self.base_url=base_url_ or os.getenv("BASE_URL")
        self.timeout=timeout or int(os.getenv("LLM_TIMEOUT",60))
        if not all([self.model,self.apikey,self.base_url]):
            raise ValueError("模型ID、API密钥和服务地址必须被提供或在.env文件中定义。")

        self.client=OpenAI(api_key=self.apikey,base_url=self.base_url,timeout=self.timeout)


    def generate(self,message:List[Dict[str,str]],temperature:float=0,output:bool=True)->str:
        """
        大模型进行思考并生成内容
        """

        print(f"{self.model}思考中...")
        try:
            response=self.client.chat.completions.create(model=self.model,messages=message,temperature=temperature,stream=True,reasoning_effort="high")

            reply=[]
            for chunk in response:
                if not chunk.choices:
                    continue
                content=chunk.choices[0].delta.content or ""
                if output:
                    print(content,end="",flush=True)
                reply.append(content)
            return "".join(reply)
        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e},重试一次")
            try:
                response = self.client.chat.completions.create(model=self.model, messages=message,
                                                               temperature=temperature, stream=True,
                                                               reasoning_effort="high")

                reply = []
                for chunk in response:
                    if not chunk.choices:
                        continue
                    content = chunk.choices[0].delta.content or ""
                    if output:
                        print(content, end="", flush=True)
                    reply.append(content)
                return "".join(reply)
            except Exception as ee:
                print(f"❌ 调用LLM API时发生错误: {e}")
                return ""



if __name__=="__main__":
    try:
        client=HelloAgentLLM()
        message=[{"role":"system","content":"你是一个私人家教，你将用宠溺、通俗的语气与用户对话，解决用户的问题"},
                 {"role":"user","content":"老师贴贴！"}]
        client.generate(message,0.3)
        print()
    except Exception as e:
        print(f"错误：{e}")
