# PDF Chat App

A Streamlit app that lets you upload any PDF and chat with it using RAG (Retrieval-Augmented Generation).

## Quick Start

```bash
# 1. Clone / unzip the project
cd pdf_chat_app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your OpenAI API key
cp .env.example .env
# Edit .env and add your key: OPENAI_API_KEY=sk-...

# 4. Run
streamlit run app.py
```

## Architecture

```
PDF Upload → Text Extraction → Chunking → Embeddings → ChromaDB
                                                          ↓
User Question → Retrieve top-5 chunks (MMR) → GPT-4o-mini → Answer + Citations
```

## Models & Config

| Component   | Choice                          | Detail                          |
|-------------|----------------------------------|----------------------------------|
| Embedding   | `text-embedding-ada-002` (OpenAI) | 1536 dimensions                 |
| LLM         | `gpt-4o-mini`                   | Fast, cost-effective             |
| Vector DB   | ChromaDB (local)                | Persisted at `./storage/`        |
| Chunking    | RecursiveCharacterTextSplitter  | 800 chars, 150 overlap           |

## Chunking Strategy

**RecursiveCharacterTextSplitter** splits on `\n\n → \n → . → space` in order,
so paragraph boundaries are respected before falling back to sentence/word splits.
- **chunk_size=800**: small enough for precise retrieval, large enough for context.
- **chunk_overlap=150**: preserves sentence context across chunk boundaries.

## Retrieval Strategy

**MMR (Maximal Marginal Relevance)** — fetches 20 candidates then re-ranks to
return 5 diverse, relevant chunks. This avoids returning near-duplicate passages
and ensures the answer draws from different parts of the document.

## Conversation History

The last 3 exchanges (6 messages) are injected into every prompt so follow-up
questions can reference earlier answers. History resets when a new PDF is uploaded.

## Known Limitations

- Single PDF per session; uploading a new PDF clears history.
- Very large PDFs (>200 pages) may be slow to embed on first upload.
- Scanned PDFs without OCR layer will yield no text.
- ChromaDB storage grows per session; manually delete `./storage/` to reset.
