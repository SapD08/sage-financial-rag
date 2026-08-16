import json
import sys
import time
import os
sys.path.insert(0, ".")
from config import OPENAI_API_KEY, OPENAI_EXTRACTION_MODEL, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

from openai import OpenAI
from neo4j import GraphDatabase

DATA_CHUNKS = "data/chunks/text_chunks.json"
CACHE_DIR = "cache/extraction"
os.makedirs(CACHE_DIR, exist_ok=True)

EXTRACTION_PROMPT = """Extract business entities and relationships from this financial text.
Return ONLY valid JSON, no explanation, no markdown.

Format:
{{
  "entities": [
    {{"type": "Company|Person|Sector|Regulation|Metric", "name": "..."}}
  ],
  "relationships": [
    {{"from": "...", "relation": "OPERATES_IN|LED_BY|COMPETES_WITH|SUBJECT_TO|REPORTS_METRIC", "to": "..."}}
  ]
}}

Rules:
- Only extract what is explicitly stated
- Keep entity names short and clean
- If nothing relevant: return {{"entities": [], "relationships": []}}

Text:
{text}
"""

client = OpenAI(api_key=OPENAI_API_KEY)
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def extract_entities(chunk_id, text):
    cache_file = f"{CACHE_DIR}/{chunk_id}.json"
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=OPENAI_EXTRACTION_MODEL,
                messages=[{"role": "user", "content": EXTRACTION_PROMPT.format(text=text[:1500])}],
                temperature=0,
                max_tokens=2000,
            )
            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            result = json.loads(raw)

            with open(cache_file, "w") as f:
                json.dump(result, f)
            return result

        except Exception as e:
            err = str(e)
            if "rate_limit" in err.lower() or "429" in err:
                wait = (attempt + 1) * 10
                print(f"  Rate limit hit : waiting {wait}s")
                time.sleep(wait)
            else:
                print(f"  Error on chunk {chunk_id}: {err[:80]}")
                break

    return {"entities": [], "relationships": []}

def load_to_neo4j(entities, relationships):
    with driver.session() as session:
        for e in entities:
            name = e.get("name", "").strip()
            etype = e.get("type", "Entity")
            if not name:
                continue
            if etype not in ("Company", "Person", "Sector", "Regulation", "Metric"):
                etype = "Entity"
            session.run(
                f"MERGE (n:{etype} {{name: $name}})",
                name=name
            )

        for r in relationships:
            from_name = r.get("from", "").strip()
            to_name   = r.get("to", "").strip()
            relation  = r.get("relation", "").strip()
            if not all([from_name, to_name, relation]):
                continue
            if relation not in ("OPERATES_IN", "LED_BY", "COMPETES_WITH", "SUBJECT_TO", "REPORTS_METRIC"):
                continue
            try:
                session.run(
                    f"MATCH (a {{name: $from_name}}), (b {{name: $to_name}}) "
                    f"MERGE (a)-[:{relation}]->(b)",
                    from_name=from_name, to_name=to_name
                )
            except:
                pass

def main(smoke_test=False):
    with open(DATA_CHUNKS) as f:
        chunks = json.load(f)

    if smoke_test:
        chunks = chunks[:20]
        print(f"[SMOKE TEST] Processing {len(chunks)} chunks")
    else:
        print(f"Processing {len(chunks)} chunks")

    stats = {"entities": 0, "relationships": 0, "skipped": 0}

    for i, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        if not text.strip():
            continue

        result = extract_entities(chunk["id"], text)
        entities = result.get("entities", [])
        rels = result.get("relationships", [])

        if entities or rels:
            load_to_neo4j(entities, rels)
            stats["entities"] += len(entities)
            stats["relationships"] += len(rels)
        else:
            stats["skipped"] += 1

        if (i + 1) % 50 == 0:
            print(f"  Chunk {i+1}/{len(chunks)} | entities={stats['entities']} rels={stats['relationships']} skipped={stats['skipped']}")

        time.sleep(0.1)

    print(f"\nDone! Stats: {stats}")

    with driver.session() as session:
        node_count = session.run("MATCH (n) RETURN count(n) as c").single()["c"]
        edge_count = session.run("MATCH ()-[r]->() RETURN count(r) as c").single()["c"]
        print(f"Neo4j: {node_count} nodes, {edge_count} edges")

    driver.close()

if __name__ == "__main__":
    smoke = "--smoke-test" in sys.argv
    main(smoke_test=smoke)