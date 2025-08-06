# import os
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import streamlit as st
# from dotenv import load_dotenv
# from langchain.vectorstores import Chroma
#
# load_dotenv()
#
# st.set_page_config(page_title="Chatbot with PDF", page_icon=":books:", layout="wide")
# st.title("Chatbot with PDF")
#
# current_dir = os.path.dirname(os.path.abspath(__file__))
# persistent_directory = os.path.join(current_dir, "db", "chroma_db")
#
# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#
# db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
#
# if not os.path.exists(persistent_directory):
#     st.write("Persistent directory does not exist. Please create it first.")
#
# text_input = st.text_input("Enter your text")
#
# retriever = db.as_retriever(
#     search_type="similarity",
#     search_kwargs={"k": 3}
# )
#
# relevant_docs = retriever.invoke(text_input)
#
# st.write("--- Retrieved Documents ---")
# for i, doc in enumerate(relevant_docs):
#     st.write(f"Document {i}: {doc.page_content}")
#     if doc.metadata:
#         st.write(f"Metadata: {doc.metadata}")



import os
from dotenv import load_dotenv
import streamlit as st

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load environment variables
load_dotenv()

# Streamlit UI
st.set_page_config(page_title="Chatbot with PDF", page_icon="📘", layout="wide")
st.title("📚 Chatbot with PDF")

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Load embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Check if vectorstore exists
if not os.path.exists(persistent_directory):
    st.error("Persistent directory does not exist. Please create it first by uploading and embedding a PDF.")
    st.stop()

# Load Chroma vectorstore
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# User question
text_input = st.text_input("🔍 Enter your question:")

if text_input:
    # Retrieve similar documents
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    relevant_docs = retriever.invoke(text_input)

    # Combine chunks
    combined_docs = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Set up Gemini for short answer
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

    prompt = PromptTemplate.from_template(
        """Answer the question based only on the following context.

Context:
{context}

Question:
{question}

Only return those part of the content that is related to the {question}, nothing else."""
    )

    qa_chain = LLMChain(llm=llm, prompt=prompt)

    # Get answer
    answer = qa_chain.run({"context": combined_docs, "question": text_input})

    # Display short answer
    st.write("### 💡 Answer:")
    st.success(answer)
