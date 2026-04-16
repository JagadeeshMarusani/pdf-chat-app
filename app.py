import streamlit as st
from utils.embeddings import create_vector_store
from utils.rag_pipeline import generate_answer
from utils.pdf_loader import load_and_chunk_pdf

st.set_page_config(page_title="Chat with your PDF", layout="wide")

st.title("📄 Chat with your PDF")

# Sidebar
with st.sidebar:
    st.header("📁 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if st.button("🗑 Clear conversation"):
        st.session_state.chat_history = []

# Session state
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Process PDF
if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        chunks = load_and_chunk_pdf(uploaded_file)
        vectordb = create_vector_store(chunks)
        retriever = vectordb.as_retriever(search_kwargs={"k": 4})

        st.session_state.vectordb = vectordb
        st.session_state.retriever = retriever

    st.success("PDF processed successfully!")

# Chat UI
query = st.chat_input("Ask a question about your PDF")

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Handle query
if query:
    if st.session_state.retriever is None:
        st.warning("Please upload a PDF first.")
    else:
        # User message
        st.session_state.chat_history.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.write(query)

        # Retrieve docs (FIXED)
        docs = st.session_state.retriever.invoke(query)

        # Generate answer
        response = generate_answer(
            query,
            docs,
            st.session_state.chat_history
        )

        answer = response["answer"]

        # Assistant response
        with st.chat_message("assistant"):
            st.write(answer)

            st.markdown("### 📚 Sources")
            for src in response["sources"]:
                st.markdown(f"- Page {src['page']} → {src['snippet']}")

        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer}
        )