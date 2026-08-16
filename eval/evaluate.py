import json
import csv
import sys
import time
import os
import re
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, ".")
from config import OPENAI_API_KEY, OPENAI_MINI_MODEL
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ── Category classification ───────────────────────────────────────────────────
SHORT_ANSWER_CATEGORIES = [
    "Table-Compare and Sort",
    "Table-Numerical Calculation",
]

LONG_ANSWER_CATEGORIES = [
    "Text Inference",
    "Text-MultiPage",
    "Text-Multipage",
    "Table-MultiPage",
    "Table-Multipage",
    "Chart-Information Extraction",
    "Chart-Time Sensitive",
    "Chart-Numerical Calculation",
]

# ── Load ground truth ─────────────────────────────────────────────────────────
def load_qrels():
    qrels = defaultdict(list)
    with open("datasets/finragbench/qrels/qrels_en.tsv") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            qrels[row["query-id"]].append(row["corpus-id"])
    return qrels

def load_questions():
    with open("datasets/finragbench/valid_questions.json") as f:
        return json.load(f)

# ── Retrieval metrics ─────────────────────────────────────────────────────────
def precision_at_k(retrieved_pages, gold_pages, k=5):
    top_k = retrieved_pages[:k]
    hits = sum(1 for p in top_k if p in gold_pages)
    return hits / k

def recall_at_k(retrieved_pages, gold_pages, k=5):
    if not gold_pages:
        return 0.0
    top_k = retrieved_pages[:k]
    hits = sum(1 for p in top_k if p in gold_pages)
    return hits / len(gold_pages)

def mean_reciprocal_rank(retrieved_pages, gold_pages):
    for i, page in enumerate(retrieved_pages):
        if page in gold_pages:
            return 1.0 / (i + 1)
    return 0.0

def chunks_to_corpus_ids(chunks):
    corpus_ids = []
    for chunk in chunks:
        doc_id = chunk.get("doc_id", "")
        page   = chunk.get("page", "")
        if doc_id and page:
            corpus_ids.append(f"{doc_id}_{page}.png")
    return corpus_ids

# ── Generation metrics ────────────────────────────────────────────────────────
def normalize_answer(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def exact_match(predicted, gold):
    return 1.0 if normalize_answer(predicted) == normalize_answer(gold) else 0.0

def f1_score(predicted, gold):
    pred_tokens = normalize_answer(predicted).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens or not gold_tokens:
        return 0.0

    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

def safe_mean(values):
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    import statistics
    return statistics.mean(valid)

def llm_score(question, answer, contexts):
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Clean UNVERIFIED tags before scoring
    clean_answer = re.sub(r'\[UNVERIFIED\]\s*', '', answer).strip()

    contexts = [c for c in contexts if c and len(c.strip()) > 10]
    if not contexts or not clean_answer:
        return {"faithfulness": 0.0, "answer_relevancy": 0.0}

    context_str = "\n\n".join(contexts[:3])[:3000]

    try:
        faith_resp = client.chat.completions.create(
            model=OPENAI_MINI_MODEL,
            messages=[{
                "role": "user",
                "content": f"""Rate how well the answer is supported by the context on a scale of 0.0 to 1.0.
Context: {context_str}
Answer: {clean_answer}
Return ONLY a number between 0.0 and 1.0. Nothing else."""
            }],
            temperature=0,
            max_tokens=5,
        )
        faithfulness = float(faith_resp.choices[0].message.content.strip())

        rel_resp = client.chat.completions.create(
            model=OPENAI_MINI_MODEL,
            messages=[{
                "role": "user",
                "content": f"""Rate how relevant the answer is to the question on a scale of 0.0 to 1.0.
Question: {question}
Answer: {clean_answer}
Return ONLY a number between 0.0 and 1.0. Nothing else."""
            }],
            temperature=0,
            max_tokens=5,
        )
        relevancy = float(rel_resp.choices[0].message.content.strip())

        return {
            "faithfulness":     min(max(faithfulness, 0.0), 1.0),
            "answer_relevancy": min(max(relevancy, 0.0), 1.0),
        }
    except Exception as e:
        print(f"    Scoring error: {e}")
        return {"faithfulness": 0.0, "answer_relevancy": 0.0}

# ── Single question evaluation ────────────────────────────────────────────────
def evaluate_question(question, config_name, retrieve_fn, qrels):
    query       = question["query"]
    query_id    = question["query_id"]
    gold_pages  = qrels.get(query_id, [])
    gold_answer = question.get("answer", "")
    category    = question.get("category", "")

    start = time.time()
    try:
        result = retrieve_fn(query)
    except Exception as e:
        print(f"    Retrieval error: {e}")
        return None
    latency = time.time() - start

    chunks        = result.get("sources", [])
    answer        = result.get("answer", "")
    retrieved_ids = chunks_to_corpus_ids(chunks)

    # Retrieval metrics — always computed
    p5  = precision_at_k(retrieved_ids, gold_pages, k=5)
    r5  = recall_at_k(retrieved_ids, gold_pages, k=5)
    mrr = mean_reciprocal_rank(retrieved_ids, gold_pages)

    # EM + F1 only for short answer categories
    if category in SHORT_ANSWER_CATEGORIES:
        em = exact_match(answer, gold_answer)
        f1 = f1_score(answer, gold_answer)
    else:
        em = None
        f1 = None

    # LLM-based metrics — always computed
    contexts = [c.get("text", "") for c in chunks]
    scores   = llm_score(query, answer, contexts)

    return {
        "query_id":         query_id,
        "query":            query,
        "category":         category,
        "config":           config_name,
        "answer":           answer,
        "gold_answer":      gold_answer,
        "gold_pages":       gold_pages,
        "retrieved_ids":    retrieved_ids,
        "precision@5":      p5,
        "recall@5":         r5,
        "mrr":              mrr,
        "exact_match":      em,
        "f1":               f1,
        "faithfulness":     scores["faithfulness"],
        "answer_relevancy": scores["answer_relevancy"],
        "latency":          round(latency, 3),
    }

# ── Main evaluation loop ──────────────────────────────────────────────────────
def run_evaluation(configs, questions, output_file="eval/results.json",
                   max_questions=None):
    qrels = load_qrels()

    # Resume support — load existing results
    if os.path.exists(output_file):
        with open(output_file) as f:
            results = json.load(f)
        done_keys = set(f"{r['query_id']}_{r['config']}" for r in results)
        print(f"Resuming — {len(results)} results already done")
    else:
        results = []
        done_keys = set()

    qs    = questions[:max_questions] if max_questions else questions
    total = len(qs) * len(configs)
    done  = len(results)

    for config_name, retrieve_fn in configs.items():
        print(f"\nEvaluating config: {config_name}")
        for q in qs:
            key = f"{q['query_id']}_{config_name}"
            if key in done_keys:
                done += 1
                continue

            print(f"  [{done+1}/{total}] {q['query'][:60]}...")
            row = evaluate_question(q, config_name, retrieve_fn, qrels)
            if row:
                results.append(row)
                done_keys.add(key)
            done += 1

            # Save after every question
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)

            time.sleep(0.3)

    print(f"\nSaved {len(results)} results to {output_file}")
    return results

# ── Results summary ───────────────────────────────────────────────────────────
def summarize(results_file="eval/results.json"):
    with open(results_file) as f:
        results = json.load(f)

    import statistics
    by_config = defaultdict(list)
    for r in results:
        by_config[r["config"]].append(r)

    print("\n" + "="*75)
    print("EVALUATION RESULTS SUMMARY")
    print("="*75)

    for config, rows in by_config.items():
        p5  = statistics.mean(r["precision@5"]      for r in rows)
        r5  = statistics.mean(r["recall@5"]         for r in rows)
        mrr = statistics.mean(r["mrr"]              for r in rows)
        em  = safe_mean(r.get("exact_match")        for r in rows)
        f1  = safe_mean(r.get("f1")                 for r in rows)
        fa  = statistics.mean(r["faithfulness"]     for r in rows)
        ar  = statistics.mean(r["answer_relevancy"] for r in rows)
        lat = statistics.mean(r["latency"]          for r in rows)

        print(f"\n{config}:")
        print(f"  Precision@5:      {p5:.3f}")
        print(f"  Recall@5:         {r5:.3f}")
        print(f"  MRR:              {mrr:.3f}")
        print(f"  Exact Match:      {f'{em:.3f}' if em is not None else 'N/A'}")
        print(f"  F1 Score:         {f'{f1:.3f}' if f1 is not None else 'N/A'}")
        print(f"  Faithfulness:     {fa:.3f}")
        print(f"  Answer Relevancy: {ar:.3f}")
        print(f"  Avg Latency:      {lat:.2f}s")
        print(f"  Questions:        {len(rows)}")

    print("\n" + "="*75)
    print("BY CATEGORY (Precision@5)")
    print("="*75)

    by_cat_config = defaultdict(lambda: defaultdict(list))
    for r in results:
        by_cat_config[r["category"]][r["config"]].append(r["precision@5"])

    configs_list = list(by_config.keys())
    header = f"{'Category':<35}" + "".join(f"{c:<15}" for c in configs_list)
    print(header)
    print("-" * len(header))

    for cat, config_data in sorted(by_cat_config.items()):
        row = f"{cat:<35}"
        for c in configs_list:
            vals = config_data.get(c, [0])
            row += f"{statistics.mean(vals):<15.3f}"
        print(row)

    print("\n" + "="*75)
    print("BY CATEGORY (F1 Score — short answer categories only)")
    print("="*75)

    by_cat_f1 = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r.get("f1") is not None:
            by_cat_f1[r["category"]][r["config"]].append(r["f1"])

    if by_cat_f1:
        print(header)
        print("-" * len(header))
        for cat, config_data in sorted(by_cat_f1.items()):
            row = f"{cat:<35}"
            for c in configs_list:
                vals = config_data.get(c, [0])
                row += f"{statistics.mean(vals):<15.3f}"
            print(row)
    else:
        print("No short answer categories in this result set.")

    print("\n" + "="*75)
    print("BY CATEGORY (Faithfulness)")
    print("="*75)

    by_cat_faith = defaultdict(lambda: defaultdict(list))
    for r in results:
        by_cat_faith[r["category"]][r["config"]].append(r["faithfulness"])

    print(header)
    print("-" * len(header))
    for cat, config_data in sorted(by_cat_faith.items()):
        row = f"{cat:<35}"
        for c in configs_list:
            vals = config_data.get(c, [0])
            row += f"{statistics.mean(vals):<15.3f}"
        print(row)

if __name__ == "__main__":
    summarize()