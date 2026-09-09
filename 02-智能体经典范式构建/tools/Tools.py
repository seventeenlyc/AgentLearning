from typing import Dict, Any, Callable
import tools.search

class ToolExecutor:
    """
    一个工具执行器，负责管理和执行工具。
    """
    def __init__(self):
        self.tools:Dict[str,Dict[str,Any]]={}

    def registerTool(self,name:str,description:str,func:Callable):
        """
        把一个函数工具注册到工具箱里
        :param name: 工具名
        :param description:工具的描述
        :param func: 工具
        """
        if not name or not func:
            return f"错误！请传入工具名与工具"
        if name in self.tools:
            print(f"警告！工具{name}已在工具箱中，将覆盖原有工具")
        self.tools[name]={"description":description,"func":func}
        print(f"工具{name}已经完成注册")
        return None

    def getTool(self,name:str)->Callable:
        """
        根据工具名字获取工具的执行函数
        """
        if name not in self.tools:
            return lambda name :f"🚫工具{name}没有注册在工具箱内，请先调用registerTool方法注册工具或者检查工具名{name}是否正确！"
        return self.tools[name]["func"]

    def getAvaliableTools(self)->str:
        """
        列出已经注册的所有工具
        """
        return "\n".join([
            f"-{name}:{info['description']}" for name,info in self.tools.items()
        ])

if __name__=="__main__":
    tool=ToolExecutor()
    tool.registerTool("Search",search.description,search.search)
    print(f"\n\n---可用工具---")
    print(tool.getAvaliableTools())
    print("\n====测试====")
    toolname="Search"
    question=input()
    tool_fuc=tool.getTool(toolname)
    observation=tool_fuc(question)
    print("--- 观察 (Observation) ---")
    print(observation)