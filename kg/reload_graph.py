import json
import os
import sys
sys.path.insert(0, ".")
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from neo4j import GraphDatabase

CACHE_DIR = "cache/extraction"
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def load_to_neo4j(entities, relationships):
    with driver.session() as session:
        for e in entities:
            name = e.get("name", "").strip()
            etype = e.get("type", "Entity")
            if not name:
                continue
            if etype not in ("Company", "Person", "Sector", "Regulation", "Metric"):
                etype = "Entity"
            session.run(f"MERGE (n:{etype} {{name: $name}})", name=name)

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

# Load all cache files
cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".json")]
print(f"Loading {len(cache_files)} cached extractions...")

for i, filename in enumerate(cache_files):
    with open(f"{CACHE_DIR}/{filename}") as f:
        result = json.load(f)
    entities = result.get("entities", [])
    rels = result.get("relationships", [])
    if entities or rels:
        load_to_neo4j(entities, rels)
    if (i + 1) % 500 == 0:
        print(f"  Loaded {i+1}/{len(cache_files)}")

# Final counts
with driver.session() as session:
    nodes = session.run("MATCH (n) RETURN count(n) as c").single()["c"]
    edges = session.run("MATCH ()-[r]->() RETURN count(r) as c").single()["c"]
    print(f"\nDone! Nodes: {nodes}, Edges: {edges}")

driver.close()