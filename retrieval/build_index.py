import json
import sys
import os
sys.path.insert(0, ".")

import chromadb
from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "finrag"
BGE_PREFIX = "Represent this sentence for searching relevant passages: "

def load_all_chunks():
    all_chunks = []
    for chunk_type in ["text", "table", "figure"]:
        path = f"data/chunks/{chunk_type}_chunks.json"
        if os.path.exists(path):
            with open(path) as f:
                chunks = json.load(f)
            print(f"  Loaded {len(chunks)} {chunk_type} chunks")
            all_chunks.extend(chunks)
    return all_chunks

def build_index():
    print("Loading chunks...")
    chunks = load_all_chunks()
    print(f"  Total: {len(chunks)} chunks")

    print(f"\nLoading embedding model: {EMBED_MODEL}...")
    model = SentenceTransformer(EMBED_MODEL)

    print("\nConnecting to ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    try:
        client.delete_collection(COLLECTION_NAME)
        print("  Deleted existing collection")
    except:
        pass

    collection = client.create_collection(COLLECTION_NAME)

    BATCH_SIZE = 50
    total = len(chunks)

    print(f"\nIndexing {total} chunks in batches of {BATCH_SIZE}...")
    for i in range(0, total, BATCH_SIZE):
        batch     = chunks[i: i + BATCH_SIZE]
        texts     = [c["text"] for c in batch]
        ids       = [c["id"] for c in batch]
        metadatas = [{
            "doc_id": c.get("doc_id", ""),
            "page":   str(c.get("page", "")),
            "type":   c.get("type", "text"),
        } for c in batch]

        # Apply BGE prefix to documents at indexing time
        prefixed_texts = [BGE_PREFIX + t for t in texts]
        embeddings = model.encode(
            prefixed_texts,
            show_progress_bar=False
        ).tolist()

        collection.add(
            documents=texts,        # store original text, not prefixed
            embeddings=embeddings,  # but embed with prefix
            metadatas=metadatas,
            ids=ids
        )

        if (i // BATCH_SIZE + 1) % 10 == 0:
            print(f"  Indexed {min(i + BATCH_SIZE, total)}/{total}")

    print(f"\nDone. Collection '{COLLECTION_NAME}' has {collection.count()} chunks.")

if __name__ == "__main__":
    build_index()