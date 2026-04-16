
def get_retriever(vectordb):
    """
    MMR (Maximal Marginal Relevance) retrieval:
    - Balances relevance + diversity so answers aren't repetitive.
    - fetch_k=20 candidates, return top k=6 after MMR re-ranking.
    - Increased k from 5→6 to give the LLM more context to work with.
    """
    return vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20},
    )