from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

chat_history = [
    SystemMessage(content="You are a helpful AI assistant."),
]

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() == "exit":
        break
    results = model.invoke(chat_history)
    chat_history.append(AIMessage(content=results.content))
    print(f"AI: {results.content}")

print("Chat history", chat_history)