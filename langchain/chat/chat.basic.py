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

# Dictionary format
# conversation = [
#     {"role": "system", "content": "You are a helpful assistant that translates English to French."},
#     {"role": "user", "content": "Translate: I love programming."},
#     {"role": "assistant", "content": "J'adore la programmation."},
#     {"role": "user", "content": "Translate: I love building applications."}
# ]

# Message objects
conversation = [
    SystemMessage("You are a helpful assistant that translates English to French."),
    HumanMessage("Translate: I love programming."),
    AIMessage("J'adore la programmation."),
    HumanMessage("Translate: I love building applications.")
]

response = agent.invoke({
    "messages": conversation
})

print(response)