from langchain.agents import create_agent
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()

# context: a dataclass to hold the context for the agent, including the thread_id and any other relevant information.
@dataclass
class Context:
    user_id: str

agent = create_agent(
    model="google_genai:gemini-2.5-flash",
    tools=[],
    checkpointer=InMemorySaver(),
    context_schema=Context,
)

config = {
    "configurable": {
        "thread_id": str(uuid7()),
    }
}

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "What's the weather in New York City today?"
    }]
}, config=config, context=Context(user_id="user_123"))

print(result["messages"][-1].content)

print("#####################################################")

# A follow-up turn on the same conversation: reuse the same thread_id to keep history

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What about tomorrow?"}]},
    config=config,
    context=Context(user_id="user_123")
)
print(result["messages"][-1].content)