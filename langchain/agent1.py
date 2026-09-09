from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain.agents import create_agent, AgentState
from langchain.tools import tool
from pydantic import BaseModel

load_dotenv()

# AgentState
class MyState(AgentState):
    user_id: str
    call_count: int

# Structured Ouput
class Answer(BaseModel):
    summary: str
    confidence: float

# Tools
@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Search results for: {query}"

# init_chat_model initializes a chat model with specified parameters such as model name, provider, temperature, timeout, max tokens, and streaming option. This model will be used by the agent to generate responses based on the input it receives.
model = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0.5,
    timeout=600,
    max_tokens=25000,
    streaming=True,
)


agent = create_agent(
    model=model, # the chat model to use
    tools=[search], # tools for the agent
    system_prompt="You are a helpful assistant. Be concise and accurate.", # system prompt for the agent
    response_format=Answer, # specify the structured output format
    state_schema=MyState # specify the state schema for the agent
)

result = agent.invoke({"messages": [{"role": "user", "content": "Summarize AI trends"}], "user_id": "user_123", "call_count": 2})
print(result["structured_response"] ) 