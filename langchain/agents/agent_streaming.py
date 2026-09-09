from langchain.agents import create_agent
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()


# Context
@dataclass
class Context:
    user_id: str


# Agent
agent = create_agent(
    model="google_genai:gemini-2.5-flash",
    tools=[],
    checkpointer=InMemorySaver(),
    context_schema=Context,
)


# Same thread_id = same conversation
config = {
    "configurable": {
        "thread_id": str(uuid7()),
    }
}


# --------------------------------------------------
# First turn - streaming
# --------------------------------------------------

print("User: What's the weather in New York City today?")

stream = agent.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "What's the weather in New York City today?",
            }
        ]
    },
    config=config,
    context=Context(user_id="user_123"),
    stream_mode="values",
)

for snapshot in stream:
    latest_message = snapshot["messages"][-1]

    if isinstance(latest_message, HumanMessage):
        print(f"User: {latest_message.content}")

    elif isinstance(latest_message, AIMessage):
        if latest_message.content:
            print(f"Agent: {latest_message.content}")

        if latest_message.tool_calls:
            print(
                f"Calling tools: "
                f"{[tc['name'] for tc in latest_message.tool_calls]}"
            )


print("#####################################################")


# --------------------------------------------------
# Follow-up turn - same conversation
# --------------------------------------------------

print("User: What about tomorrow?")

stream = agent.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "What about tomorrow?",
            }
        ]
    },
    config=config,
    context=Context(user_id="user_123"),
    stream_mode="values",
)

for snapshot in stream:
    latest_message = snapshot["messages"][-1]

    if isinstance(latest_message, HumanMessage):
        print(f"User: {latest_message.content}")

    elif isinstance(latest_message, AIMessage):
        if latest_message.content:
            print(f"Agent: {latest_message.content}")

        if latest_message.tool_calls:
            print(
                f"Calling tools: "
                f"{[tc['name'] for tc in latest_message.tool_calls]}"
            )