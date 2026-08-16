import sys
import json
sys.path.insert(0, ".")
from config import (NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
                    GROQ_API_KEY, GROQ_MODEL, OPENAI_API_KEY,
                    OPENAI_MINI_MODEL, EMBED_MODEL)
from neo4j import GraphDatabase
from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
groq_client = Groq(api_key=GROQ_API_KEY)
embed_model = SentenceTransformer(EMBED_MODEL)
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_collection("finrag")

BGE_PREFIX = "Represent this sentence for searching relevant passages: "

ENTITY_PROMPT = """Extract named entities from this query. Return ONLY a JSON list of strings.
Query: {query}
Return format: ["Entity One", "Entity Two"]
If none found: []"""

def extract_entities(query):
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": ENTITY_PROMPT.format(query=query)}],
            temperature=0,
            max_tokens=100,
        )
        return json.loads(response.choices[0].message.content.strip())
    except:
        return []

def get_node_ids(entity_names):
    node_ids = []
    with driver.session() as session:
        for name in entity_names:
            result = session.run("""
                MATCH (n)
                WHERE toLower(n.name) CONTAINS toLower($name)
                RETURN elementId(n) AS eid
                LIMIT 5
            """, name=name)
            for rec in result:
                node_ids.append(rec["eid"])
    return node_ids

def personalized_pagerank(seed_node_ids, top_k=20):
    if not seed_node_ids:
        return []
    with driver.session() as session:
        result = session.run("""
            MATCH (n)
            WHERE elementId(n) IN $seed_ids
            WITH collect(n) AS sourceNodes
            CALL gds.pageRank.stream("finrag_graph", {
                dampingFactor: 0.85,
                maxIterations: 20,
                sourceNodes: sourceNodes
            })
            YIELD nodeId, score
            WHERE score > 0.15
            RETURN gds.util.asNode(nodeId).name AS name,
                   labels(gds.util.asNode(nodeId))[0] AS type,
                   score
            ORDER BY score DESC
            LIMIT $top_k
        """, seed_ids=seed_node_ids, top_k=top_k)
        return [(rec["name"], rec["type"], rec["score"]) for rec in result]

def retrieve_chunks_for_ranked_nodes(query, ranked_nodes, k=5):
    """Use PageRank-ranked entities to guide ChromaDB retrieval."""
    top_entities = [name for name, ntype, score in ranked_nodes[:10]]
    entity_str = " ".join(top_entities)
    enriched_query = f"{query} {entity_str}"
    embedding = embed_model.encode([BGE_PREFIX + enriched_query]).tolist()

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

def generate_answer(query, ranked_nodes, chunks):
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    entity_context = "\n".join([
        f"{name} ({ntype}, relevance={score:.3f})"
        for name, ntype, score in ranked_nodes
    ]) if ranked_nodes else "No ranked entities found."

    doc_str = "\n\n".join([c["text"] for c in chunks])

    response = client.chat.completions.create(
        model=OPENAI_MINI_MODEL,
        messages=[{
            "role": "user",
            "content": f"""Answer the question using the ranked entities and document excerpts below.

Ranked Relevant Entities (from knowledge graph):
{entity_context}

Document Excerpts:
{doc_str}

Question: {query}

Answer:"""
        }],
        temperature=0,
        max_tokens=500,
    )
    return response.choices[0].message.content

def run(query, k=5):
    entities = extract_entities(query)
    seed_ids = get_node_ids(entities) if entities else []
    ranked_nodes = personalized_pagerank(seed_ids) if seed_ids else []
    chunks = retrieve_chunks_for_ranked_nodes(query, ranked_nodes, k=k)
    answer = generate_answer(query, ranked_nodes, chunks)

    return {
        "answer":        answer,
        "sources":       chunks,
        "ranked_nodes":  ranked_nodes[:10],
        "seed_entities": entities,
        "config":        "hippo_rag"
    }

if __name__ == "__main__":
    query = "What sectors does Howden operate in and what regulations apply?"
    result = run(query)
    print(f"Query: {query}")
    print(f"Seed entities: {result['seed_entities']}")
    print(f"\nAnswer: {result['answer']}")
    print(f"\nTop ranked nodes:")
    for name, ntype, score in result["ranked_nodes"]:
        print(f"  [{ntype}] {name} — {score:.3f}")
    print(f"\nSources:")
    for s in result["sources"]:
        print(f"  [{s['type'].upper()}] {s['doc_id']} p{s['page']}")