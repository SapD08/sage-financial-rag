import json
import csv
from collections import defaultdict

# Load data
with open("datasets/finragbench/queries/queries_en.json") as f:
    queries = json.load(f)

with open("datasets/finragbench/qrels/qrels_en.tsv") as f:
    reader = csv.DictReader(f, delimiter="\t")
    qrels = list(reader)

# Build: doc_id → set of query_ids that reference it
doc_to_queries = defaultdict(set)
for row in qrels:
    corpus_id = row["corpus-id"]
    doc_id = "_".join(corpus_id.split("_")[:-1])
    doc_to_queries[doc_id].add(row["query-id"])

# Build: query_id → category
query_to_cat = {q["query-id"]: q["category"] for q in queries}

# Modality scoring: count distinct modality types per doc
def get_modalities(doc_id, query_ids):
    cats = [query_to_cat.get(qid, "") for qid in query_ids]
    modalities = set()
    for c in cats:
        if "Table" in c:
            modalities.add("table")
        if "Chart" in c:
            modalities.add("chart")
        if "Text" in c:
            modalities.add("text")
    return len(modalities)

# Filter: English-only doc IDs (no Chinese characters)
def is_english(doc_id):
    return all(ord(c) < 128 for c in doc_id)

# Score each document
doc_scores = {}
for doc_id, query_ids in doc_to_queries.items():
    if not is_english(doc_id):
        continue
    n_questions = len(query_ids)
    n_modalities = get_modalities(doc_id, query_ids)
    score = n_questions * n_modalities
    doc_scores[doc_id] = {
        "score": score,
        "questions": n_questions,
        "modalities": n_modalities,
        "query_ids": list(query_ids)
    }

# Sort and pick top 25
ranked = sorted(doc_scores.items(), key=lambda x: x[1]["score"], reverse=True)

print(f"Total English documents: {len(ranked)}")
print(f"\nTop 25 selected documents:")
print(f"{'Rank':<5} {'Score':<8} {'Qs':<5} {'Mod':<5} Document ID")
print(" ")

selected_docs = []
for i, (doc_id, info) in enumerate(ranked[:25], 1):
    print(f"{i:<5} {info['score']:<8} {info['questions']:<5} {info['modalities']:<5} {doc_id}")
    selected_docs.append(doc_id)

# Count valid questions (all gold docs in selected set)
query_to_gold_docs = defaultdict(set)
for row in qrels:
    corpus_id = row["corpus-id"]
    doc_id = "_".join(corpus_id.split("_")[:-1])
    query_to_gold_docs[row["query-id"]].add(doc_id)

valid_questions = []
for q in queries:
    qid = q["query-id"]
    gold_docs = query_to_gold_docs[qid]
    if gold_docs.issubset(set(selected_docs)):
        valid_questions.append(q)

print(f"\nValid questions after subset filter: {len(valid_questions)}")
print(f"\nSaving selected_docs.json")

import json
with open("datasets/finragbench/selected_docs.json", "w") as f:
    json.dump({
        "selected_docs": selected_docs,
        "valid_question_count": len(valid_questions)
    }, f, indent=2)

print("Done")