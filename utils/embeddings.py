from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os


def create_vector_store(chunks):
    texts = [c["text"] for c in chunks]
    metadatas = [{"page": c["page"]} for c in chunks]

    embeddings = OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_API_BASE"),
    )

    vectordb = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory="./storage",
    )

    return vectordb