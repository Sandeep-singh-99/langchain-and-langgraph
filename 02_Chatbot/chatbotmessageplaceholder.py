from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

chat_template = ChatPromptTemplate.from_messages([
    ('system', 'You are a very helpful customer service assistant.'),
    MessagesPlaceholder(variable_name="chat_history"),
    ('human', '{query}')
])

chat_history = []
with open("chatbot_history.txt") as file:
    chat_history.extend(file.readlines())

prompt = chat_template.invoke({
    "chat_history": chat_history,
    "query": "Where is my refund?"
})

print(prompt)