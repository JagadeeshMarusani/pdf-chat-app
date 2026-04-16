
import streamlit as st
from utils.embeddings import create_vector_store
from utils.rag_pipeline import generate_answer
from utils.pdf_loader import load_and_chunk_pdf
from utils.retriever import get_retriever

st.set_page_config(page_title="Chat with your PDF", layout="wide")

st.title("📄 Chat with your PDF")

# Sidebar
with st.sidebar:
    st.header("📁 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if st.button("🗑 Clear conversation"):
        st.session_state.chat_history = []
        st.rerun()

# Session state
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

# ✅ Only re-process if a NEW file is uploaded (avoid re-processing on every rerun)
if uploaded_file is not None:
    if st.session_state.last_uploaded_file != uploaded_file.name:
        with st.spinner("Processing PDF... Please wait."):
            chunks = load_and_chunk_pdf(uploaded_file)
            vectordb = create_vector_store(chunks)

            # ✅ Use MMR retriever for better, diverse results
            retriever = get_retriever(vectordb)

            st.session_state.vectordb = vectordb
            st.session_state.retriever = retriever
            st.session_state.last_uploaded_file = uploaded_file.name
            st.session_state.chat_history = []  # Reset chat on new PDF

        st.success(f"✅ PDF processed successfully! ({len(chunks)} chunks indexed)")

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Sources"):
                for src in msg["sources"]:
                    st.markdown(f"**[{src['ref']}] Page {src['page']}**")
                    st.caption(src["snippet"])

# Chat input
query = st.chat_input("Ask a question about your PDF")

# Handle query
if query:
    if st.session_state.retriever is None:
        st.warning("⚠️ Please upload a PDF first.")
    else:
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # ✅ Retrieve relevant docs using MMR
                docs = st.session_state.retriever.invoke(query)

                # ✅ Generate answer
                response = generate_answer(
                    query,
                    docs,
                    st.session_state.chat_history
                )

            answer = response["answer"]
            sources = response["sources"]

            st.write(answer)

            if sources:
                with st.expander("📚 Sources"):
                    for src in sources:
                        st.markdown(f"**[{src['ref']}] Page {src['page']}**")
                        st.caption(src["snippet"])

        # Save assistant message with sources
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })