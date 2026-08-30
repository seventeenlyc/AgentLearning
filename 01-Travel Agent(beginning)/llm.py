from openai import OpenAI
class LLMClient:
    """
    一个用于调用如何兼容openai接口的LLM
    """
    def __init__(self,model:str,base_url:str,api_key:str):
        self.model = model
        self.client=OpenAI(api_key=api_key,base_url=base_url,)

    def generate(self,prompt:str,system_prompt:str):
        print(f"正在调用{self.model}...")
        try:
            message=[
                {"role":"system","content":system_prompt},
                {"role":"user","content":prompt}
            ]
            response=self.client.chat.completions.create(model=self.model,messages=message,stream=False)
            answer=response.choices[0].message.content
            print("大模型响应成功！")
            return answer
        except Exception as e:
            print(f"调用LLM API时发生错误: {e}")
            return "错误:调用语言模型服务时出错。"

if __name__=="__main__":
    llm=LLMClient("hf.co/HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive:Q4_K_M","http://localhost:11434","ollama")
    llm.generate("hi","")