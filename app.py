import streamlit as st
from src.rag import ask_rag


st.set_page_config(
    page_title="Insurance Claims Copilot",
    page_icon="📄",
    layout="wide"
)

st.title("Insurance Claims Copilot")

st.write(
    "RAG-powered application for querying insurance "
    "policies, claims guidelines and SOPs."
)

question = st.text_input(
    "Ask a question about the insurance documents:"
)

if question:
    st.subheader("Your Question")
    st.write(question)

    st.subheader("Response")
with st.spinner("Searching insurance documents and generating answer..."):

    answer, documents = ask_rag(question)

st.subheader("Response")

st.write(answer)

st.subheader("Sources")

for i, document in enumerate(documents, start=1):

    source = document.metadata.get(
        "source",
        "Unknown source"
    )

    with st.expander(f"Source {i}: {source}"):

        st.write(document.page_content)