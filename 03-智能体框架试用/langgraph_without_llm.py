from typing import TypedDict

from langgraph.graph import StateGraph, START, END


class NumberState(TypedDict):
    number: int

def check_number(state):
    return {}

def route_number(state):
    return "even" if state["number"]%2==0 else "odd"

def event_number(state):
    return {
        "number":state["number"]//2
    }

def odd_number(state):
    return {
        "number":state["number"]*3+1
    }

def check_finish(state):
    return {}

def check_continue(state):
    return "continue" if state['number']!=1 else "finish"

def finish(state):
    return {
        "number":state["number"]
    }

graph=StateGraph(NumberState)

graph.add_node("check",check_number)
graph.add_node("odd_node",odd_number)
graph.add_node("even_node",event_number)
graph.add_node("check_finish",check_finish)
graph.add_node("finish",finish)

graph.add_edge(START,"check")
graph.add_conditional_edges("check",route_number,{
    "odd":"odd_node",
    "even":"even_node",
})

graph.add_edge("odd_node","check_finish")
graph.add_edge("even_node","check_finish")

graph.add_conditional_edges("check_finish",check_continue,{
    "continue":"check",
    "finish":"finish",
})

graph.add_edge("finish",END)

res=graph.compile()

print(res.invoke({"number":547854}))
print(res.invoke({"number":2**999}))