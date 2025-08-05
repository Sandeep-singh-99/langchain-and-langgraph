from qa_engine import load_and_split, save_to_chroma, get_qa_chain
from langchain.callbacks.base import BaseCallbackHandler
from typing import Any
import streamlit as st
import tempfile
import os

class StreamHandler(BaseCallbackHandler):
    def __init__(self):
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs: Any):
        self.text += token
        st.session_state.output_placeholder.markdown(self.text)

st.set_page_config(page_title="Notes chatbot with Gemini", layout="wide")
st.title("📄 Chat with Your Notes")

uploaded_file = st.file_uploader("Upload a PDF, Markdown, or Text file", type=["pdf", "md", "txt"])

if uploaded_file:
    extension = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.success("✅ File uploaded and being processed...")

    chunks = load_and_split(tmp_path)
    vectordb = save_to_chroma(chunks)

    stream_handler = StreamHandler()
    qa_chain = get_qa_chain(vectordb, callbacks=[stream_handler])
    st.session_state.qa_chain = qa_chain

    st.success("✅ Ready to ask questions!")

if "qa_chain" in st.session_state:
    questions = st.text_input("Ask a question about the document:")
    if questions:
        st.session_state.output_placeholder = st.empty()
        st.session_state.qa_chain.invoke({"query": questions})
