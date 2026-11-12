import streamlit as st
import sys
import json
import time
import re
import statistics
from collections import defaultdict
sys.path.insert(0, ".")

st.set_page_config(
    page_title="SAGE — Multimodal Agentic RAG Benchmark",
    layout="wide"
)

# ── Load results
@st.cache_data
def load_results():
    with open("eval/results.json") as f:
        return json.load(f)

@st.cache_data
def compute_summary(results):
    by_config = defaultdict(list)
    for r in results:
        by_config[r["config"]].append(r)

    def safe_mean(values):
        valid = [v for v in values if v is not None]
        return round(statistics.mean(valid), 3) if valid else None

    config_order = ["vanilla_rag", "graph_rag", "hippo_rag", "sage"]
    summary = {}
    for config in config_order:
        rows = by_config.get(config, [])
        if not rows:
            continue
        summary[config] = {
            "precision@5":      safe_mean(r["precision@5"]      for r in rows),
            "recall@5":         safe_mean(r["recall@5"]         for r in rows),
            "mrr":              safe_mean(r["mrr"]              for r in rows),
            "exact_match":      safe_mean(r.get("exact_match")  for r in rows),
            "f1":               safe_mean(r.get("f1")           for r in rows),
            "faithfulness":     safe_mean(r["faithfulness"]     for r in rows),
            "answer_relevancy": safe_mean(r["answer_relevancy"] for r in rows),
            "latency":          safe_mean(r["latency"]          for r in rows),
            "questions":        len(rows),
        }
    return summary

# ── Page header ───────────────────────────────────────────────────────────────
st.title("SAGE — Structured Agentic Graph-Enhanced RAG with Self-Correction")
st.markdown("""
**Multimodal Agentic RAG for Business Intelligence: GraphRAG, HippoRAG, and Vector Retrieval with Self-Correction**  
Dataset: FinRAGBench-V (EMNLP 2025) | 18 documents | 232 questions | 928 evaluation runs  
Saptarshi Das MTech Computer Science & Data Processing, IIT Kharagpur
""")
st.divider()

# ── Navigation tabs
tab_demo, tab_results, tab_category, tab_about = st.tabs([
    "Live Demo",
    "Benchmark Results",
    "Category Analysis",
    "About"
])


with tab_demo:
    st.markdown("### Run a Query")
    st.markdown("Select configs, enter a question, and compare responses side by side.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        run_vanilla = st.checkbox("Vanilla RAG", value=True)
    with col2:
        run_graph = st.checkbox("GraphRAG", value=True)
    with col3:
        run_hippo = st.checkbox("HippoRAG", value=True)
    with col4:
        run_sage = st.checkbox("SAGE", value=True)

    SAMPLE_QUESTIONS = [
        "",
        "According to Howden Joinery Group Plc Annual Reports 2023, which is higher between cash at bank in 2022 and 2023?",
        "What regulations is Ambac subject to in the UK?",
        "Which companies compete with Ambac in the insurance sector?",
        "What sectors does Howden operate in?",
        "According to the SRCE 2022 Form 10-K, which year had a lower net income?",
        "What methods did the auditors use to ensure effectiveness of internal controls for Ambac in 2021?",
        "According to Craneware plc 2022 annual report, which region contributed the highest revenue?",
    ]

    sample = st.selectbox("Sample questions:", SAMPLE_QUESTIONS)
    query = st.text_area(
        "Enter your question:",
        value=sample,
        height=80,
        placeholder="Type a financial question about the documents..."
    )

    run_button = st.button("Run Query", type="primary", disabled=not query.strip())

    if run_button and query.strip():
        selected = []
        if run_vanilla: selected.append("Vanilla RAG")
        if run_graph:   selected.append("GraphRAG")
        if run_hippo:   selected.append("HippoRAG")
        if run_sage:    selected.append("SAGE")

        if not selected:
            st.warning("Please select at least one config.")
        else:
            with st.spinner("Loading models (first run may take 30s)..."):
                from retrieval.vanilla_rag import run as vanilla_run
                from retrieval.graph_retriever import run as graph_run
                from retrieval.hippo_retriever import run as hippo_run
                from agent.sage import run as sage_run

            config_map = {
                "Vanilla RAG": vanilla_run,
                "GraphRAG":    graph_run,
                "HippoRAG":    hippo_run,
                "SAGE":        sage_run,
            }

            cols = st.columns(len(selected))

            for col, config_name in zip(cols, selected):
                run_fn = config_map[config_name]
                with col:
                    st.markdown(f"**{config_name}**")
                    with st.spinner(f"Running..."):
                        start = time.time()
                        try:
                            result = run_fn(query)
                            latency = time.time() - start

                            answer = result.get("answer", "")
                            clean_answer = re.sub(r'\[UNVERIFIED\]\s*', '', answer).strip()

                            st.markdown("**Answer:**")
                            st.info(clean_answer)
                            st.caption(f"Latency: {latency:.2f}s")

                            if config_name == "SAGE":
                                flagged     = result.get("flagged", 0)
                                corrections = result.get("corrections", 0)
                                st.caption(f"Self-corrections: {corrections} | Flagged claims: {flagged}")

                            with st.expander("Sources"):
                                for s in result.get("sources", []):
                                    st.markdown(
                                        f"- [{s.get('type','text').upper()}] "
                                        f"`{s.get('doc_id','')}` p{s.get('page','')}"
                                    )

                            if config_name == "GraphRAG":
                                paths = result.get("graph_paths", [])
                                if paths:
                                    with st.expander("Graph Paths"):
                                        for p in paths[:8]:
                                            st.markdown(f"- `{p}`")

                            if config_name == "HippoRAG":
                                nodes = result.get("ranked_nodes", [])
                                if nodes:
                                    with st.expander("Ranked Entities"):
                                        for name, ntype, score in nodes[:5]:
                                            st.markdown(f"- **{name}** ({ntype}) {score:.3f}")

                        except Exception as e:
                            st.error(f"Error: {str(e)[:200]}")


with tab_results:
    st.markdown("### Overall Benchmark Results")
    st.markdown("232 questions x 4 configs = 928 evaluation runs. Fully automatic scoring via qrels.")

    try:
        results = load_results()
        summary = compute_summary(results)
        import pandas as pd

        config_labels = {
            "vanilla_rag": "Vanilla RAG",
            "graph_rag":   "GraphRAG",
            "hippo_rag":   "HippoRAG",
            "sage":        "SAGE",
        }

        metrics_display = [
            ("Precision@5",      "precision@5"),
            ("Recall@5",         "recall@5"),
            ("MRR",              "mrr"),
            ("Exact Match",      "exact_match"),
            ("F1 Score",         "f1"),
            ("Faithfulness",     "faithfulness"),
            ("Answer Relevancy", "answer_relevancy"),
            ("Avg Latency (s)",  "latency"),
        ]

        rows = []
        for metric_name, key in metrics_display:
            row = {"Metric": metric_name}
            best_val = -1
            best_config = None
            for config in ["vanilla_rag", "graph_rag", "hippo_rag", "sage"]:
                val = summary.get(config, {}).get(key)
                if val is not None and key != "latency" and val > best_val:
                    best_val = val
                    best_config = config
            for config in ["vanilla_rag", "graph_rag", "hippo_rag", "sage"]:
                val = summary.get(config, {}).get(key)
                label = config_labels[config]
                if val is not None:
                    marker = " *" if config == best_config and key != "latency" else ""
                    row[label] = f"{val:.3f}{marker}"
                else:
                    row[label] = "N/A"
            rows.append(row)

        df = pd.DataFrame(rows).set_index("Metric")
        st.dataframe(df, use_container_width=True)
        st.caption("* = best value for that metric (excluding latency)")

        st.markdown("### Winners by Metric")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Best Retrieval (P@5)", "HippoRAG", "0.103")
            st.metric("Best Recall", "HippoRAG", "0.466")
        with c2:
            st.metric("Best Ranking (MRR)", "SAGE", "0.377")
            st.metric("Best F1", "SAGE", "0.297")
        with c3:
            st.metric("Best Faithfulness", "HippoRAG", "0.745")
            st.metric("Best Answer Relevancy", "GraphRAG", "0.952")
        with c4:
            st.metric("Fastest", "Vanilla RAG", "1.68s")
            st.metric("Best Exact Match", "SAGE", "0.170")

    except FileNotFoundError:
        st.warning("eval/results.json not found. Run evaluation first.")


with tab_category:
    st.markdown("### Performance by Question Category")

    try:
        results = load_results()
        import pandas as pd

        config_labels = {
            "vanilla_rag": "Vanilla RAG",
            "graph_rag":   "GraphRAG",
            "hippo_rag":   "HippoRAG",
            "sage":        "SAGE",
        }
        config_order = ["vanilla_rag", "graph_rag", "hippo_rag", "sage"]

        metric_choice = st.selectbox(
            "Select metric:",
            ["Precision@5", "Faithfulness", "F1 Score (short answer only)",
             "Answer Relevancy", "MRR"]
        )

        metric_key_map = {
            "Precision@5":                  "precision@5",
            "Faithfulness":                 "faithfulness",
            "F1 Score (short answer only)": "f1",
            "Answer Relevancy":             "answer_relevancy",
            "MRR":                          "mrr",
        }
        key = metric_key_map[metric_choice]

        by_cat = defaultdict(lambda: defaultdict(list))
        for r in results:
            val = r.get(key)
            if val is not None:
                by_cat[r["category"]][r["config"]].append(val)

        rows = []
        for cat in sorted(by_cat.keys()):
            row = {"Category": cat}
            for config in config_order:
                vals = by_cat[cat].get(config, [])
                row[config_labels[config]] = round(statistics.mean(vals), 3) if vals else None
            rows.append(row)

        df = pd.DataFrame(rows).set_index("Category")
        st.dataframe(df, use_container_width=True)
        st.bar_chart(df)

        st.markdown("### Category Descriptions")
        st.markdown("""
| Category | Description | Expected Winner |
|----------|-------------|----------------|
| Table-Compare and Sort | Compare values across rows/columns | Vanilla RAG / SAGE |
| Table-Numerical Calculation | Compute revenue, margins, EPS | GraphRAG |
| Table-MultiPage | Financial data spanning multiple pages | HippoRAG |
| Text Inference | Narrative passages, MD&A, risk factors | Vanilla RAG |
| Text-MultiPage | Multi-page text reasoning | HippoRAG |
| Chart-Information Extraction | Extract data from charts | HippoRAG |
| Chart-Time Sensitive | Time-series chart questions | HippoRAG |
""")

    except FileNotFoundError:
        st.warning("eval/results.json not found. Run evaluation first.")

# ════════════════════════════════════════════════════════════════════════════
# TAB 4: ABOUT
# ════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("### Project Overview")
    st.markdown("""
**SAGE (Structured Agentic Graph-Enhanced RAG with Self-Correction)** is a controlled benchmark 
comparing four RAG architectures on real financial documents from FinRAGBench-V (EMNLP 2025).

### Architecture

**Config 1 — Vanilla RAG**
- ChromaDB vector store with BAAI/bge-large-en-v1.5 embeddings (1024-dim)
- Smart metadata routing: table chunks for financial queries, figure chunks for chart queries
- Fastest config at 1.68s average latency

**Config 2 — GraphRAG**
- Neo4j Knowledge Graph: 11,052 nodes, 23,077 edges
- Entity extraction via Groq LLM → 2-hop Cypher traversal with fuzzy matching
- Graph-discovered entities enrich ChromaDB query for better retrieval
- Best answer relevancy: 0.952

**Config 3 — HippoRAG**
- Same Neo4j KG as GraphRAG
- Personalized PageRank via Neo4j GDS plugin (damping=0.85, maxIterations=20)
- PageRank propagates relevance through entire graph — finds indirect connections
- Best retrieval: Precision@5 (0.103), Recall@5 (0.466), Faithfulness (0.745)

**Config 4 — SAGE**
- Parallel retrieval: runs all 3 configs simultaneously
- Context fusion: deduplicates chunks by doc_id + page
- Reranking: BAAI/bge-reranker-large cross-encoder scores all fused chunks
- Self-correction: claim-level LLM judge (threshold 0.6), re-retrieval for borderline claims (0.3-0.6)
- Best ranking and factual accuracy: MRR (0.377), Exact Match (0.170), F1 (0.297)

### Dataset
- **FinRAGBench-V** (EMNLP 2025) — financial RAG benchmark with visual citation
- 51,219 English pages from 1,105 financial documents
- 539 human-annotated QA pairs across 7 question categories
- Subset used: 18 documents, 232 valid questions after principled document-first selection

### Tech Stack
- **Embeddings:** BAAI/bge-large-en-v1.5 (local, free, 1024-dim)
- **Reranker:** BAAI/bge-reranker-large (local, free, 560M parameters)
- **LLM:** GPT-4o Mini (answer generation + scoring) + Groq Llama-3.3-70B (entity extraction)
- **Vector Store:** ChromaDB (persistent, 6,181 chunks)
- **Graph DB:** Neo4j + GDS plugin (Docker, 11,052 nodes, 23,077 edges)
- **KG Extraction:** GPT-4o Mini (5,256 chunks processed)
- **Vision:** GPT-4o (table + chart extraction, ~$1.50 total)
- **Framework:** Pure Python, no LangChain agent framework needed

### Evaluation Methodology
- **Corpus selection:** Documents scored by (questions x modalities), selected before question filtering
- **Question filtering:** Only questions where ALL gold pages are in selected documents
- **Scoring:** Fully automatic via qrels.tsv — zero manual annotation
- **Short answer extraction:** GPT-4o Mini post-processing for Table categories before EM/F1 scoring
- **Metrics:** Precision@5, Recall@5, MRR, Exact Match, F1, Faithfulness, Answer Relevancy, Latency
""")

    st.divider()
    st.markdown("*Saptarshi Das MTech Computer Science & Data Processing, IIT Kharagpur | FinRAGBench-V (EMNLP 2025)*")