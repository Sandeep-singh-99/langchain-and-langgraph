from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator
from langchain.messages import SystemMessage
from langchain.messages import ToolMessage
from dotenv import load_dotenv

load_dotenv()


model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

"""
#####################
Define Tools
#####################
"""

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two integers and returns the result."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Adds two integers and returns the result."""
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """Subtracts the second integer from the first and returns the result."""
    return a - b

@tool
def divide(a: int, b: int) -> float:
    """Divides the first integer by the second and returns the result."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b

"""
#####################
Augment the llm with tools
#####################
"""

tools = [multiply, add, subtract, divide]   
tool_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)


"""
#####################
Define State
#####################
"""

class MessageState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


"""
#####################
Define model node
#####################
"""

def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""
    return {
        "messages": [
            model_with_tools.invoke([
                SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")
            ] + state["messages"])
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


"""
#####################
Define tool node
#####################
"""

def tool_node(state: dict):
    """Performs the tool call"""

    results = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tool_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        results.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    
    return {"messages": results}

"""
#####################
Step 5: Define logic to determine whether to end
#####################
"""

from typing import Literal
from langgraph.graph import StateGraph, START, END


"""
#####################
Conditional edge function to route to the tool node or end based upon whether the LLM made a tool call
#####################
"""

def should_continue(state: MessageState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    """ If the LLM makes a tool call, then perform an action"""
    if last_message.tool_calls:
        return "tool_node"
    
    """Otherwise, we stop (reply to the user)"""
    return END

""" Step 6: Build agent """

"""Build workflow"""

agent_builder = StateGraph(MessageState)

"""Add nodes"""

agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

"""Add edges to connect the nodes"""
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ['tool_node', END])

""" Compile the agent """
agent = agent_builder.compile()


from IPython.display import Image, display

"""Show the agent """

display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

"""Invoke"""
from langchain.messages import HumanMessage

messages = [HumanMessage(content="Add 3 and 4.")]
messages = agent.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()