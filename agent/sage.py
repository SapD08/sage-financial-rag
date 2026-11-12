"""
SAGE — Structured Agentic Graph-Enhanced RAG
Config 4: Parallel retrieval + Context fusion + Reranking + Self-correction
Reranker: BAAI/bge-reranker-large (domain-aligned with BGE embedding model)
"""
import sys
import time
import re
sys.path.insert(0, ".")

from retrieval.vanilla_rag import vanilla_retrieve
from retrieval.graph_retriever import run as graph_run
from retrieval.hippo_retriever import run as hippo_run
from config import OPENAI_API_KEY, OPENAI_MINI_MODEL

#Reranker 
reranker = None

def get_reranker():
    global reranker
    if reranker is None:
        print("Loading BGE reranker")
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder("models/bge-reranker-large")
        print("BGE reranker loaded.")
    return reranker

#Context fusion 
def fuse_chunks(chunk_lists):
    """Merge chunks from multiple retrievers, deduplicate by doc_id + page."""
    seen = set()
    fused = []
    for chunks in chunk_lists:
        for chunk in chunks:
            key = f"{chunk.get('doc_id','')}_{chunk.get('page','')}"
            if key not in seen:
                seen.add(key)
                fused.append(chunk)
    return fused

#Reranking 
def rerank_chunks(query, chunks, top_k=5):
    """Score all chunks against query using BGE reranker, return top_k."""
    if not chunks:
        return []
    ranker = get_reranker()
    pairs  = [(query, chunk["text"]) for chunk in chunks]
    scores = ranker.predict(pairs)
    scored = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]

# Self-correction 
def self_correct(query, draft_answer, context_chunks):
    """Verify each claim in draft answer against context. Flag unsupported claims."""
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    context = "\n\n".join([c["text"] for c in context_chunks])[:4000]

    sentences = re.split(r"(?<=[.!?])\s+", draft_answer.strip())
    claims = [s.strip() for s in sentences if len(s.split()) >= 5]

    if not claims:
        return draft_answer, 0, 0

    verified = []
    flagged_count = 0
    correction_count = 0

    for claim in claims:
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MINI_MODEL,
                messages=[{
                    "role": "user",
                    "content": f"""Does the context support this claim? Score 0.0-1.0.
Context: {context[:2000]}
Claim: {claim}
Return ONLY a number 0.0-1.0."""
                }],
                temperature=0,
                max_tokens=5,
            )
            score = float(resp.choices[0].message.content.strip())
        except:
            score = 0.5

        if score >= 0.6:
            verified.append(claim)
        elif score >= 0.3:
            try:
                extra_chunks = vanilla_retrieve(claim, k=3)
                extra_context = "\n".join([c["text"] for c in extra_chunks])
                resp2 = client.chat.completions.create(
                    model=OPENAI_MINI_MODEL,
                    messages=[{
                        "role": "user",
                        "content": f"""Does the context support this claim? Score 0.0-1.0.
Context: {extra_context[:2000]}
Claim: {claim}
Return ONLY a number 0.0-1.0."""
                    }],
                    temperature=0,
                    max_tokens=5,
                )
                re_score = float(resp2.choices[0].message.content.strip())
                if re_score >= 0.6:
                    verified.append(claim)
                    correction_count += 1
                else:
                    verified.append(f"[UNVERIFIED] {claim}")
                    flagged_count += 1
            except:
                verified.append(claim)
        else:
            verified.append(f"[UNVERIFIED] {claim}")
            flagged_count += 1

    return " ".join(verified), flagged_count, correction_count

#Answer generation 
def generate_answer(query, chunks):
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    context = "\n\n".join([c["text"] for c in chunks])

    resp = client.chat.completions.create(
        model=OPENAI_MINI_MODEL,
        messages=[{
            "role": "user",
            "content": f"""Answer the question using only the context below.
Be concise and specific. Use exact numbers and dates from the context.

Context:
{context}

Question: {query}

Answer:"""
        }],
        temperature=0,
        max_tokens=500,
    )
    return resp.choices[0].message.content

#Main SAGE pipeline
def run(query, k=5, use_self_correction=True):
    start = time.time()

    # Step 1: Parallel retrieval
    vanilla_chunks = vanilla_retrieve(query, k=k)
    graph_chunks   = graph_run(query, k=k).get("sources", [])
    hippo_chunks   = hippo_run(query, k=k).get("sources", [])

    # Step 2: Context fusion
    all_chunks = fuse_chunks([vanilla_chunks, graph_chunks, hippo_chunks])

    # Step 3: Reranking with BGE reranker
    top_chunks = rerank_chunks(query, all_chunks, top_k=k)

    # Step 4: Generate draft answer
    draft_answer = generate_answer(query, top_chunks)

    # Step 5: Self-correction
    flagged = corrections = 0
    if use_self_correction:
        final_answer, flagged, corrections = self_correct(
            query, draft_answer, top_chunks
        )
    else:
        final_answer = draft_answer

    return {
        "answer":       final_answer,
        "draft_answer": draft_answer,
        "sources":      top_chunks,
        "flagged":      flagged,
        "corrections":  corrections,
        "latency":      round(time.time() - start, 3),
        "config":       "sage",
    }

# ── CLI test 
if __name__ == "__main__":
    query = "According to Howden Joinery Group Plc Annual Reports 2023, which is higher between cash at bank in 2022 and 2023?"
    print(f"Query: {query}\n")
    result = run(query)
    print(f"Answer:\n{result['answer']}")
    print(f"\nFlagged claims: {result['flagged']}")
    print(f"Corrections made: {result['corrections']}")
    print(f"Latency: {result['latency']}s")
    print(f"\nSources:")
    for s in result["sources"]:
        print(f"  [{s['type'].upper()}] {s['doc_id']} p{s['page']}")