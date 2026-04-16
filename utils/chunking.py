from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(pages: list[dict]) -> list[dict]:
    """
    General-purpose chunking (works for ANY PDF):
    - Preserves paragraph structure
    - Avoids very small/noisy chunks
    - Keeps semantic meaning intact
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []

    for page in pages:
        splits = splitter.split_text(page["text"])

        for chunk in splits:
            chunk = chunk.strip()

            # ✅ Skip noise
            if len(chunk) < 50:
                continue

            chunks.append({
                "text": chunk,
                "page": page["page"]
            })

    return chunks