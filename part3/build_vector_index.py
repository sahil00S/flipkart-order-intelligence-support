from pathlib import Path
import json
import re

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


KB_DIR = Path("knowledge_base")
INDEX_DIR = Path("part3/vector_index")

INDEX_DIR.mkdir(parents=True, exist_ok=True)


def sentence_chunk(text):
    """
    Split a policy document into sentence-level chunks.
    """
    text = re.sub(r"\s+", " ", text).strip()

    sentences = re.split(r"(?<=[.!?])\s+", text)

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


documents = []
chunks = []


# ------------------------------------------------------------
# Load policy documents
# ------------------------------------------------------------

for path in sorted(KB_DIR.glob("*.md")):
    text = path.read_text(encoding="utf-8")

    document_id = path.stem

    documents.append(
        {
            "document_id": document_id,
            "source": path.name,
        }
    )

    document_chunks = sentence_chunk(text)

    for chunk_id, chunk_text in enumerate(document_chunks):
        chunks.append(
            {
                "chunk_id": len(chunks),
                "document_id": document_id,
                "source": path.name,
                "chunk_index": chunk_id,
                "text": chunk_text,
            }
        )


print("Documents:", len(documents))
print("Sentence chunks:", len(chunks))


# ------------------------------------------------------------
# Local sentence-transformer embeddings
# ------------------------------------------------------------

print()
print("Loading local embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

texts = [chunk["text"] for chunk in chunks]

embeddings = embedding_model.encode(
    texts,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True,
)

embeddings = embeddings.astype("float32")

print("Embedding shape:", embeddings.shape)


# ------------------------------------------------------------
# FAISS index
# ------------------------------------------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print("FAISS vectors:", index.ntotal)


# ------------------------------------------------------------
# Save index
# ------------------------------------------------------------

faiss.write_index(
    index,
    str(INDEX_DIR / "policy.index"),
)


# ------------------------------------------------------------
# Save document/chunk metadata
# ------------------------------------------------------------

with open(
    INDEX_DIR / "chunks.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        chunks,
        f,
        indent=2,
        ensure_ascii=False,
    )


with open(
    INDEX_DIR / "documents.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        documents,
        f,
        indent=2,
        ensure_ascii=False,
    )


print()
print("=== Vector Index Built ===")
print("Index:", INDEX_DIR / "policy.index")
print("Chunks:", INDEX_DIR / "chunks.json")
print("Documents:", INDEX_DIR / "documents.json")