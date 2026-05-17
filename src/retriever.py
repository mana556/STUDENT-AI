def get_retriever(vector_db, search_k=20):
    return vector_db.as_retriever(search_kwargs={"k": search_k})