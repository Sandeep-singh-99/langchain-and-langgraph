from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import streamlit as st
import os

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="Chat Model Basic Conversation", page_icon=":robot:")
st.title("Chat Model Basic Conversation with Google")

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

messages = [
    SystemMessage(content="solve the following math problems."),
    HumanMessage(content="what is 81 divided by 9?"),
    AIMessage(content="only answer no other context."),
]

result = model.invoke(messages)

st.write(result.content)

# messages = [
#     SystemMessage(content="solve the following math problems."),
#     HumanMessage(content="what is 81 divided by 9?"),
#     AIMessage(content="81 divided by 9 is 9."),
#     # HumanMessage(content="what is 10 times 5?"),
# ]

# result = model.invoke(messages)
# st.success(result.content)