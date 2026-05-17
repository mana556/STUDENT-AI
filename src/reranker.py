import numpy as np


def _cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def rerank_documents(query, docs, embeddings, top_k=5):
    if not docs:
        return []

    query_embedding = embeddings.embed_query(query)
    doc_texts = [doc.page_content for doc in docs]
    doc_embeddings = embeddings.embed_documents(doc_texts)

    scored_docs = [(_cosine_similarity(query_embedding, doc_vector), doc) for doc, doc_vector in zip(docs, doc_embeddings)]
    scored_docs.sort(key=lambda item: item[0], reverse=True)

    return [doc for _, doc in scored_docs[:top_k]]
