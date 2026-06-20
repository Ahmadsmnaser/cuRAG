import re
from collections import Counter

import numpy as np
import curag


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def build_vocabulary(texts: list[str]) -> dict[str, int]:
    vocab = {}

    for text in texts:
        for token in tokenize(text):
            if token not in vocab:
                vocab[token] = len(vocab)

    return vocab


def embed_text(text: str, vocab: dict[str, int]) -> np.ndarray:
    vector = np.zeros(len(vocab), dtype=np.float32)
    counts = Counter(tokenize(text))

    for token, count in counts.items():
        if token in vocab:
            vector[vocab[token]] = float(count)

    return vector


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vectors / norms


def main():
    documents = [
        "CUDA shared memory is fast on-chip memory used by threads inside the same block.",
        "cuRAG is a CUDA vector search library for retrieval augmented generation systems.",
        "Cosine similarity measures how close two normalized vectors are by using their dot product.",
        "Top-k search returns the highest scoring documents for a query.",
        "Python bindings allow a C++ CUDA library to be used directly from Python.",
        "RAG combines document retrieval with a language model to answer questions using context.",
        "GPU batching improves throughput by processing multiple queries together.",
    ]

    questions = [
        "What is CUDA shared memory?",
        "How does RAG answer questions?",
        "Why use batch search on GPU?",
    ]

    # Build vocabulary from both documents and demo questions
    # so the toy embedding can represent the query words too.
    vocab = build_vocabulary(documents + questions)
    dim = len(vocab)

    corpus_embeddings = np.array(
        [embed_text(doc, vocab) for doc in documents],
        dtype=np.float32,
    )
    corpus_embeddings = normalize(corpus_embeddings).astype(np.float32)

    index = curag.Index(dim)
    index.build(corpus_embeddings)

    for question in questions:
        query_embedding = embed_text(question, vocab).reshape(1, -1)
        query_embedding = normalize(query_embedding).astype(np.float32)[0]

        result = index.search(query_embedding, k=2)

        print("=" * 80)
        print("Question:")
        print(question)
        print()
        print("Top retrieved documents:")

        for rank, doc_index in enumerate(result["indices"], start=1):
            score = result["values"][rank - 1]
            print(f"{rank}. score={score:.4f} | {documents[doc_index]}")

    print("=" * 80)


if __name__ == "__main__":
    main()