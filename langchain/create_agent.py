from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}"

agent = create_agent(
    model="groq:openai/gpt-oss-120b",
    tools=[get_weather],
    system_prompt="You are a helpful assistant."
)

result = agent.invoke({
    "messages": "user",
    "content": "What's the weather in Delhi?"
})

print(result['messages'][-1].content_blocks)