from src.reranker import rerank_documents

def generate_answer(query, retriever, llm, embeddings=None):
    docs = retriever.invoke(query)
    if isinstance(docs, dict):
        docs = docs.get("output", [])

    if embeddings is not None:
        docs = rerank_documents(query, docs, embeddings, top_k=5)
    else:
        docs = docs[:5]

    # Limit total context size to prevent 413 errors
    max_context_chars = 2500
    context_parts = []
    total_chars = 0
    for doc in docs:
        doc_text = doc.page_content[:800]
        if total_chars + len(doc_text) > max_context_chars:
            break
        context_parts.append(doc_text)
        total_chars += len(doc_text)
    
    context = "\n".join(context_parts)

    prompt = f"""
    Answer the question based on the context below:

    Context:
    {context}

    Question:
    {query}
    """

    response = llm.predict(prompt)

    return response, docs