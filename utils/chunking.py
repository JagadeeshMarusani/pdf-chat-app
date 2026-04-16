from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(pages: list[dict]) -> list[dict]:
    """
    Split each page's text into overlapping chunks.
    Chunking strategy:
      - chunk_size=800 chars  → small enough for precise retrieval
      - chunk_overlap=150 chars → preserves sentence context across boundaries
    Returns list of dicts: [{"text": str, "page": int}, ...]
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for page in pages:
        splits = splitter.split_text(page["text"])
        for chunk in splits:
            chunks.append({"text": chunk, "page": page["page"]})
    return chunks
