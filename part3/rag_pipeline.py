from pathlib import Path
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


INDEX_DIR = Path("part3/vector_index")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class PolicyRAG:
    def __init__(self):
        self.index = faiss.read_index(
            str(INDEX_DIR / "policy.index")
        )

        with open(
            INDEX_DIR / "chunks.json",
            "r",
            encoding="utf-8",
        ) as f:
            self.chunks = json.load(f)

        with open(
            INDEX_DIR / "documents.json",
            "r",
            encoding="utf-8",
        ) as f:
            self.documents = json.load(f)

        self.embedding_model = SentenceTransformer(
            MODEL_NAME
        )

    def retrieve(self, query, k=3):
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = self.index.search(
            query_embedding,
            k,
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):
            chunk = self.chunks[int(index)]

            results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "document_id": chunk["document_id"],
                    "source": chunk["source"],
                    "text": chunk["text"],
                    "score": float(score),
                }
            )

        return results


# ------------------------------------------------------------
# Retrieval evaluation
# ------------------------------------------------------------

evaluation_queries = [
    {
        "query": "How many days can an apparel product be returned?",
        "relevant_documents": {"return_windows"},
    },
    {
        "query": "How long does a COD refund take?",
        "relevant_documents": {"cod_refund"},
    },
    {
        "query": "What is the standard delivery time?",
        "relevant_documents": {"delivery_sla"},
    },
    {
        "query": "When is reverse pickup available?",
        "relevant_documents": {"reverse_pickup"},
    },
    {
        "query": "Can a customer request a replacement?",
        "relevant_documents": {"replacement_policy"},
    },
    {
        "query": "What happens for a damaged product?",
        "relevant_documents": {"damaged_product"},
    },
]


def precision_at_k(retrieved_documents, relevant_documents, k=3):
    retrieved = retrieved_documents[:k]

    relevant_retrieved = sum(
        1
        for document_id in retrieved
        if document_id in relevant_documents
    )

    return relevant_retrieved / k


def recall_at_k(retrieved_documents, relevant_documents, k=3):
    retrieved = set(retrieved_documents[:k])

    relevant_retrieved = len(
        retrieved.intersection(relevant_documents)
    )

    return relevant_retrieved / len(relevant_documents)


if __name__ == "__main__":

    rag = PolicyRAG()

    print("=== RAG Retrieval Evaluation ===")
    print()

    precision_scores = []
    recall_scores = []

    for item in evaluation_queries:

        query = item["query"]
        relevant_documents = item["relevant_documents"]

        results = rag.retrieve(query, k=3)

        retrieved_documents = [
            result["document_id"]
            for result in results
        ]

        precision = precision_at_k(
            retrieved_documents,
            relevant_documents,
            k=3,
        )

        recall = recall_at_k(
            retrieved_documents,
            relevant_documents,
            k=3,
        )

        precision_scores.append(precision)
        recall_scores.append(recall)

        print("Query:", query)
        print("Expected:", sorted(relevant_documents))
        print("Retrieved:", retrieved_documents)

        print("Top 3 results:")

        for result in results:
            print(
                f"  {result['source']} | "
                f"score={result['score']:.4f}"
            )
            print(
                f"    {result['text']}"
            )

        print(
            f"Precision@3: {precision:.4f}"
        )

        print(
            f"Recall@3: {recall:.4f}"
        )

        print("-" * 70)

    mean_precision = float(
        np.mean(precision_scores)
    )

    mean_recall = float(
        np.mean(recall_scores)
    )

    print()
    print("=== Overall Retrieval Metrics ===")
    print(
        "Mean Precision@3:",
        round(mean_precision, 4),
    )
    print(
        "Mean Recall@3:",
        round(mean_recall, 4),
    )