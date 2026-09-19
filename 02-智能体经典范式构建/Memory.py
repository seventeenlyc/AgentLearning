from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentState:
    """一次完整用户任务的状态。"""

    task_id: str = ""
    task: str = ""
    plan: List[Dict[str, Any]] = field(default_factory=list)

    current_step: int = 0
    completed_steps: List[Dict[str, Any]] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)
    observations: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)

    final_answer: Optional[str] = None
    reflection: Optional[Dict[str, Any]] = None

    def record_step(
        self,
        task: str,
        *,
        status: str,
        observation: Any = None,
        result: Any = None,
        tool: Optional[str] = None,
        action_input: Any = None,
        error: Any = None,
    ) -> Dict[str, Any]:
        """记录一个子任务结果，并同步更新摘要字段。"""

        self.current_step += 1
        entry = {
            "step": self.current_step,
            "task": task,
            "status": status,
            "observation": observation,
            "result": result,
        }
        if tool is not None:
            entry["tool"] = tool
        if action_input is not None:
            entry["action_input"] = action_input
        if error is not None:
            entry["error"] = error

        self.history.append(entry)

        if status in {"success", "completed"}:
            self.completed_steps.append(entry.copy())       #copy()引用传递
            if observation is not None:
                self.observations.append(
                    {
                        "step": self.current_step,
                        "task": task,
                        "content": observation,
                    }
                )
        else:
            self.errors.append(
                {
                    "step": self.current_step,
                    "task": task,
                    "error": error if error is not None else result,
                }
            )

        return entry

    def get_reflection_context(self) -> Dict[str, Any]:
        """返回给 Reflection 的最小上下文，不暴露完整执行 history。"""

        return deepcopy(        #deepcopy()值传递，不会影响原对象
            {
                "task_id": self.task_id,
                "task": self.task,
                "plan": self.plan,
                "completed_steps": self.completed_steps,
                "observations": self.observations,
                "errors": self.errors,
                "final_answer": self.final_answer,
            }
        )


class AgentMemory:
    """按 task_id 管理多个任务的 AgentState。"""

    def __init__(self) -> None:
        self.states: Dict[str, AgentState] = {}

    def create(
        self,
        task_id: str,
        task: str = "",
        plan: Optional[List[Dict[str, Any]]] = None,
    ) -> AgentState:
        if not task_id:
            raise ValueError("task_id 不能为空")
        if task_id in self.states:
            raise ValueError(f"task_id 已存在: {task_id}")

        state = AgentState(task_id=task_id, task=task, plan=plan or [])
        self.states[task_id] = state
        return state

    def get(self, task_id: str) -> Optional[AgentState]:
        return self.states.get(task_id)

    def get_or_create(self, task_id: str, task: str = "") -> AgentState:
        state = self.get(task_id)
        if state is not None :
            return state
        else:
            return self.create(task_id, task)

    def remove(self, task_id: str) -> Optional[AgentState]:
        return self.states.pop(task_id, None)
