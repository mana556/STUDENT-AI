def generate_answer(query, retriever, llm):
    docs = retriever.get_relevant_documents(query)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Answer the question based on the context below:

    Context:
    {context}

    Question:
    {query}
    """

    response = llm.predict(prompt)

    return response, docs