import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer


# ======================
# CHUNK TEXT
# ======================

def chunk_text(text, chunk_size=500):

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):

        chunk = " ".join(words[i:i + chunk_size])

        chunks.append(chunk)

    return chunks


# ======================
# BUILD FAISS INDEX
# ======================

def build_faiss_index(chunks):

    vectorizer = TfidfVectorizer()

    embeddings = vectorizer.fit_transform(chunks).toarray()

    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return {
        "index": index,
        "vectorizer": vectorizer,
        "embeddings": embeddings
    }, chunks


# ======================
# RAG SEARCH
# ======================

def rag_scientific_answer(question, rag_index, chunks, top_k=3):

    index = rag_index["index"]
    vectorizer = rag_index["vectorizer"]

    question_embedding = vectorizer.transform([question]).toarray()

    question_embedding = question_embedding.astype("float32")

    distances, indices = index.search(question_embedding, top_k)

    retrieved_chunks = []

    for idx in indices[0]:

        if idx < len(chunks):

            retrieved_chunks.append(chunks[idx])

    context = "\n\n".join(retrieved_chunks)

    answer = f"""
## Scientific RAG Analysis

### Question
{question}

### Retrieved Scientific Context
{context[:4000]}

### AI Interpretation
The uploaded scientific document contains relevant information connected to the query above.
Review the retrieved context for biological observations, telemetry indicators,
environmental conditions, and scientific findings.
"""

    return answer