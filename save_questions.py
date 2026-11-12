import json
import csv
from collections import defaultdict

with open("datasets/finragbench/queries/queries_en.json") as f:
    queries = json.load(f)

with open("datasets/finragbench/qrels/qrels_en.tsv") as f:
    reader = csv.DictReader(f, delimiter="\t")
    qrels = list(reader)

with open("datasets/finragbench/selected_docs.json") as f:
    selected = json.load(f)

selected_docs = selected["selected_docs"]

query_to_gold_docs = defaultdict(set)
for row in qrels:
    corpus_id = row["corpus-id"]
    doc_id = "_".join(corpus_id.split("_")[:-1])
    query_to_gold_docs[row["query-id"]].add(doc_id)

query_to_gold_pages = defaultdict(list)
for row in qrels:
    query_to_gold_pages[row["query-id"]].append(row["corpus-id"])

valid_questions = []
for q in queries:
    qid = q["query-id"]
    gold_docs = query_to_gold_docs[qid]
    if gold_docs.issubset(set(selected_docs)):
        valid_questions.append({
            "query_id":   qid,
            "query":      q["query"],
            "answer":     q["answer"],
            "category":   q["category"],
            "gold_pages": query_to_gold_pages[qid]
        })

with open("datasets/finragbench/valid_questions.json", "w") as f:
    json.dump(valid_questions, f, indent=2)

from collections import Counter
cats = Counter(q["category"] for q in valid_questions)
print(f"Total valid questions: {len(valid_questions)}")
print("\nBy category:")
for cat, count in cats.most_common():
    print(f"  {cat}: {count}")