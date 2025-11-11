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
    SystemMessage(content="You are a friendly assistant.")
]

query = st.text_input("Ask a question:")

if query:
    messages.append(HumanMessage(content=query))

    res = model.invoke(messages)

    messages.append(AIMessage(content=res.content))

    st.success(res.content)

    st.write("------- Chat History -------")
    for message in messages:
        st.write(f"{message.type.capitalize()}: {message.content}")