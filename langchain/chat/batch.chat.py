from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0.5,
    timeout=600,
    max_tokens=25000,
    streaming=True,
)


agent = create_agent(
    model=model,
    tools=[],
)

# Batch
"""
agent.batch() expects each input to be an agent state dictionary, not a plain string.
"""
responses = agent.batch([
    {
        "messages": [
            {"role": "user", "content": "Why do parrots have colorful feathers?"}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "How do airplanes fly?"}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "What is quantum computing?"}
        ]
    }
])

for response in responses:
    print(response)

# direct model batch
responses = model.batch([
    "Why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "What is quantum computing?"
])

for response in responses:
    print(response.content)