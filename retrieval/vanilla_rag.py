import sys
sys.path.insert(0, ".")
import chromadb
from sentence_transformers import SentenceTransformer
from config import GROQ_API_KEY, GROQ_MODEL, EMBED_MODEL

model = SentenceTransformer(EMBED_MODEL)
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("finrag")

FINANCIAL_KW = ["revenue", "margin", "profit", "eps", "ebitda",
                "earnings", "income", "loss", "premium", "ratio",
                "growth", "crore", "million", "billion", "%"]
VISUAL_KW = ["chart", "figure", "graph", "trend", "bar", "pie", "line"]

BGE_PREFIX = "Represent this sentence for searching relevant passages: "

def get_filter(query):
    q = query.lower()
    if any(kw in q for kw in VISUAL_KW):
        return {"type": "figure"}
    if any(kw in q for kw in FINANCIAL_KW):
        return {"type": "table"}
    return None

def vanilla_retrieve(query, k=5):
    prefixed_query = BGE_PREFIX + query
    embedding = model.encode([prefixed_query]).tolist()
    filter_dict = get_filter(query)

    if filter_dict:
        results = collection.query(
            query_embeddings=embedding,
            n_results=k,
            where=filter_dict
        )
        if len(results["documents"][0]) < 2:
            results = collection.query(
                query_embeddings=embedding,
                n_results=k
            )
    else:
        results = collection.query(
            query_embeddings=embedding,
            n_results=k
        )

    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({
            "text":   doc,
            "doc_id": meta.get("doc_id", ""),
            "page":   meta.get("page", ""),
            "type":   meta.get("type", "text"),
        })
    return chunks

def generate_answer(query, chunks):
    from openai import OpenAI
    from config import OPENAI_API_KEY, OPENAI_MINI_MODEL
    client = OpenAI(api_key=OPENAI_API_KEY)
    context = "\n\n".join([c["text"] for c in chunks])
    response = client.chat.completions.create(
        model=OPENAI_MINI_MODEL,
        messages=[{
            "role": "user",
            "content": f"Answer the question using only the context below.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        }],
        temperature=0,
        max_tokens=500,
    )
    return response.choices[0].message.content

def run(query, k=5):
    chunks = vanilla_retrieve(query, k)
    answer = generate_answer(query, chunks)
    return {
        "answer":  answer,
        "sources": chunks,
        "config":  "vanilla_rag"
    }

if __name__ == "__main__":
    query = "What was Ambac's gross premium written in 2023?"
    result = run(query)
    print(f"Query: {query}")
    print(f"\nAnswer: {result['answer']}")
    print(f"\nSources:")
    for s in result["sources"]:
        print(f"  [{s['type'].upper()}] {s['doc_id']} p{s['page']}")