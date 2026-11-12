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

ENTITY_PROMPT = """Extract named entities from this query. Return ONLY a JSON list of entity name strings.
Include company names, person names, regulation names, sector names.
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

def traverse_graph(entity_name, hops=2):
    contexts = []
    related_entities = []
    with driver.session() as session:
        result = session.run("""
            MATCH (n)-[r]-(m)
            WHERE toLower(n.name) CONTAINS toLower($name)
            WITH n, r, m,
                 CASE type(r)
                   WHEN 'SUBJECT_TO' THEN 1
                   WHEN 'COMPETES_WITH' THEN 2
                   WHEN 'OPERATES_IN' THEN 3
                   WHEN 'REPORTS_METRIC' THEN 4
                   WHEN 'LED_BY' THEN 5
                   ELSE 6 END AS priority
            ORDER BY priority
            RETURN n.name AS src, type(r) AS rel, m.name AS dst
            LIMIT 30
        """, name=entity_name)
        for rec in result:
            contexts.append(f"{rec['src']} -[{rec['rel']}]-> {rec['dst']}")
            related_entities.append(rec['dst'])
            related_entities.append(rec['src'])

        if hops >= 2:
            result = session.run("""
                MATCH (n)-[r1]-(m)-[r2]-(k)
                WHERE toLower(n.name) CONTAINS toLower($name)
                AND k.name <> n.name
                RETURN n.name AS src, type(r1) AS rel1,
                       m.name AS mid, type(r2) AS rel2, k.name AS dst
                LIMIT 20
            """, name=entity_name)
            for rec in result:
                contexts.append(
                    f"{rec['src']} -[{rec['rel1']}]-> {rec['mid']} -[{rec['rel2']}]-> {rec['dst']}"
                )
                related_entities.extend([rec['mid'], rec['dst']])

    return contexts, list(set(related_entities))

def retrieve_chunks_for_entities(query, related_entities, k=5):
    """Use graph-discovered entities to guide ChromaDB retrieval."""
    entity_str = " ".join(related_entities[:10])
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

def generate_answer(query, graph_context, chunks):
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    graph_str = "\n".join(graph_context[:20]) if graph_context else "No graph context found."
    doc_str = "\n\n".join([c["text"] for c in chunks])

    response = client.chat.completions.create(
        model=OPENAI_MINI_MODEL,
        messages=[{
            "role": "user",
            "content": f"""Answer the question using the knowledge graph context and document excerpts below.

Knowledge Graph Context (entity relationships):
{graph_str}

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
    all_graph_contexts = []
    all_related_entities = []

    for entity in entities:
        contexts, related = traverse_graph(entity)
        all_graph_contexts.extend(contexts)
        all_related_entities.extend(related)

    seen = set()
    unique_contexts = []
    for c in all_graph_contexts:
        if c not in seen:
            seen.add(c)
            unique_contexts.append(c)

    chunks = retrieve_chunks_for_entities(query, all_related_entities, k=k)
    answer = generate_answer(query, unique_contexts, chunks)

    return {
        "answer":      answer,
        "sources":     chunks,
        "graph_paths": unique_contexts[:10],
        "config":      "graph_rag"
    }

if __name__ == "__main__":
    query = "What regulations is Ambac subject to in the UK?"
    result = run(query)
    print(f"Query: {query}")
    print(f"\nAnswer: {result['answer']}")
    print(f"\nGraph paths:")
    for p in result["graph_paths"]:
        print(f"  {p}")
    print(f"\nSources:")
    for s in result["sources"]:
        print(f"  [{s['type'].upper()}] {s['doc_id']} p{s['page']}")