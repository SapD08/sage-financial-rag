"""
Post-process existing results to extract short answers for Table categories.
Recomputes F1 and Exact Match without re-running retrievers.
"""
import json
import sys
import re
sys.path.insert(0, ".")
from config import OPENAI_API_KEY, OPENAI_MINI_MODEL
from openai import OpenAI

client = OpenAI(api_key=OPENAI_API_KEY)

SHORT_ANSWER_CATEGORIES = [
    "Table-Compare and Sort",
    "Table-Numerical Calculation",
]

def extract_short_answer(long_answer, question):
    """Extract just the key fact from a long answer."""
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MINI_MODEL,
            messages=[{
                "role": "user",
                "content": f"""Extract ONLY the direct answer to the question from the text below.
Return 1-6 words maximum. No explanation. No sentences. Just the key fact, number, date, or name.

Question: {question}
Text: {long_answer}

Direct answer (1-6 words only):"""
            }],
            temperature=0,
            max_tokens=20,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  Extraction error: {e}")
        return long_answer

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

def process():
    with open("eval/results.json") as f:
        results = json.load(f)

    print(f"Loaded {len(results)} results")

    short_count = sum(1 for r in results if r["category"] in SHORT_ANSWER_CATEGORIES)
    print(f"Short answer results to process: {short_count}")

    updated = 0
    for i, result in enumerate(results):
        if result["category"] not in SHORT_ANSWER_CATEGORIES:
            continue

        query       = result["query"]
        long_answer = result["answer"]
        gold_answer = result["gold_answer"]

        print(f"  [{updated+1}/{short_count}] {query[:50]}...")

        short_answer = extract_short_answer(long_answer, query)
        print(f"    Gold:      {gold_answer}")
        print(f"    Extracted: {short_answer}")

        em = exact_match(short_answer, gold_answer)
        f1 = f1_score(short_answer, gold_answer)

        print(f"    EM={em:.1f} F1={f1:.3f}")

        results[i]["short_answer"] = short_answer
        results[i]["exact_match"]  = em
        results[i]["f1"]           = f1

        updated += 1

        # Save after every update
        with open("eval/results.json", "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nDone. Updated {updated} results.")
    print("Saved to eval/results.json")

if __name__ == "__main__":
    process()