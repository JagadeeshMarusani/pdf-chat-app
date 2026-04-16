def get_retriever(vectordb):
    """
    MMR (Maximal Marginal Relevance) retrieval:
    - Balances relevance + diversity so answers aren't repetitive.
    - fetch_k=20 candidates, return top k=5 after MMR re-ranking.
    """
    return vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20},
    )
