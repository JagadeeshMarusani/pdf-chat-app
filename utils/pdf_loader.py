from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import os


def load_and_chunk_pdf(uploaded_file):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    # Load PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # ✅ Larger chunks = more context per retrieval = better answers
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,       # Increased from 500 → keeps full paragraphs
        chunk_overlap=200,     # Increased overlap → no lost context at boundaries
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)

    results = []

    for doc in chunks:
        text = doc.page_content.strip()

        # Skip very small/noisy chunks
        if len(text) < 30:
            continue

        results.append({
            "text": text,
            "page": doc.metadata.get("page", 0)
        })

    # Cleanup temp file
    try:
        os.unlink(file_path)
    except Exception:
        pass

    return results