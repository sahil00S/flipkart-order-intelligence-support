from pathlib import Path
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


INDEX_DIR = Path("part3/vector_index")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class PolicyRetriever:

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

        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL
        )

    def retrieve(self, query: str, k: int = 3):

        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        # Retrieve extra chunks so we can return
        # unique documents.
        search_k = min(k * 5, self.index.ntotal)

        scores, indices = self.index.search(
            query_embedding,
            search_k,
        )

        results = []
        seen_documents = set()

        for score, index in zip(
            scores[0],
            indices[0],
        ):
            if index < 0:
                continue

            chunk = self.chunks[int(index)]
            document_id = chunk["document_id"]

            if document_id in seen_documents:
                continue

            seen_documents.add(document_id)

            results.append(
                {
                    "document_id": document_id,
                    "source": chunk["source"],
                    "text": chunk["text"],
                    "score": float(score),
                }
            )

            if len(results) == k:
                break

        return results


if __name__ == "__main__":

    retriever = PolicyRetriever()

    query = "How long does a COD refund take?"

    results = retriever.retrieve(
        query,
        k=3,
    )

    print("=== Retriever Test ===")

    for result in results:
        print()
        print("Document:", result["document_id"])
        print("Source:", result["source"])
        print("Score:", round(result["score"], 4))
        print("Text:", result["text"])