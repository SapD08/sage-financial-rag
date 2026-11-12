import streamlit as st
import sys
import json
import time
import re
import statistics
from collections import defaultdict
import altair as alt
import pandas as pd
sys.path.insert(0, ".")

st.set_page_config(
    page_title="SAGE — Multimodal Agentic RAG Benchmark",
    layout="wide"
)

CONFIG_ORDER = ["vanilla_rag", "graph_rag", "hippo_rag", "sage"]
CONFIG_LABELS = {
    "vanilla_rag": "Vanilla RAG",
    "graph_rag":   "GraphRAG",
    "hippo_rag":   "HippoRAG",
    "sage":        "SAGE",
}
CONFIG_COLORS = {
    "Vanilla RAG": "#d99400",
    "GraphRAG":    "#199e70",
    "HippoRAG":    "#d9622f",
    "SAGE":        "#256abf",
}
INK = "#1c1f26"
INK_MUTED = "#6b6a65"
RULE = "#d9dfd2"
ACCENT = "#2f4a3a"

st.markdown(f"""
<style>
.block-container {{ padding-top: 3rem; max-width: 1200px; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stMarkdownContainer"] h3, [data-testid="stMarkdownContainer"] h4 {{
  font-family: 'Source Serif 4', Georgia, serif; font-weight: 600; letter-spacing: -0.01em;
}}

.hero-eyebrow {{
  font-size: 0.75rem; letter-spacing: 0.18em; text-transform: uppercase;
  color: {INK_MUTED}; margin-bottom: 0.6rem;
}}
.hero-title {{
  font-family: 'Source Serif 4', Georgia, serif; font-weight: 600;
  font-size: 3.4rem; line-height: 1.05; color: {INK}; margin: 0;
}}
.hero-subtitle {{
  font-family: 'Source Serif 4', Georgia, serif; font-weight: 500;
  font-size: 1.35rem; color: {ACCENT}; margin: 0.5rem 0 1rem 0;
}}
.hero-rule {{ width: 56px; height: 2px; background: #b08d57; margin: 1.2rem 0; border: 0; }}
.hero-lede {{ color: #3d3f45; font-size: 1rem; max-width: 760px; line-height: 1.6; }}
.hero-author {{ color: {INK_MUTED}; font-size: 0.9rem; margin-top: 0.8rem; }}
.hero-author b {{ color: {INK}; font-weight: 600; }}

.stat-row {{
  display: grid; grid-template-columns: repeat(4, 1fr);
  border-top: 1px solid {RULE}; border-bottom: 1px solid {RULE};
  margin: 2rem 0 1.5rem 0;
}}
.stat {{ padding: 1.1rem 0.25rem; }}
.stat + .stat {{ border-left: 1px solid {RULE}; padding-left: 1.25rem; }}
.stat-value {{
  font-family: 'Source Serif 4', Georgia, serif; font-size: 2rem;
  font-weight: 600; color: {INK}; line-height: 1.1;
}}
.stat-label {{
  font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase;
  color: {INK_MUTED}; margin-top: 0.3rem;
}}

.section-kicker {{
  font-size: 0.72rem; letter-spacing: 0.16em; text-transform: uppercase;
  color: #b08d57; margin: 0.5rem 0 0.2rem 0; font-weight: 600;
}}

.winner-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.9rem; }}
.winner {{
  background: #fbfcfa; border: 1px solid {RULE}; border-radius: 6px;
  padding: 1rem 1.1rem;
}}
.winner-metric {{
  font-size: 0.7rem; letter-spacing: 0.12em; text-transform: uppercase; color: {INK_MUTED};
}}
.winner-name {{
  font-family: 'Source Serif 4', Georgia, serif; font-size: 1.3rem;
  font-weight: 600; color: {INK}; margin-top: 0.35rem;
}}
.winner-value {{ font-size: 0.9rem; color: #3d3f45; margin-top: 0.1rem; font-variant-numeric: tabular-nums; }}

.dot {{
  display: inline-block; width: 9px; height: 9px; border-radius: 50%;
  margin-right: 0.5rem; vertical-align: middle; position: relative; top: -1px;
}}
.card-title {{
  font-family: 'Source Serif 4', Georgia, serif; font-size: 1.2rem;
  font-weight: 600; color: {INK};
}}
.card-tag {{ font-size: 0.8rem; color: {INK_MUTED}; margin: 0.15rem 0 0.4rem 0; }}

.stTabs [data-baseweb="tab-list"] {{ gap: 2rem; border-bottom: 1px solid {RULE}; }}
.stTabs [data-baseweb="tab"] {{ padding: 0.75rem 0; font-weight: 500; }}

.footer {{
  border-top: 1px solid {RULE}; margin-top: 3rem; padding-top: 1.2rem;
  color: {INK_MUTED}; font-size: 0.82rem; display: flex; justify-content: space-between;
}}

@media (max-width: 800px) {{
  .stat-row, .winner-grid {{ grid-template-columns: repeat(2, 1fr); }}
  .hero-title {{ font-size: 2.4rem; }}
}}
</style>
""", unsafe_allow_html=True)


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

    summary = {}
    for config in CONFIG_ORDER:
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


def dot(label):
    return f'<span class="dot" style="background:{CONFIG_COLORS[label]}"></span>'


def kicker(text):
    st.markdown(f'<div class="section-kicker">{text}</div>', unsafe_allow_html=True)


st.markdown(f"""
<div class="hero-eyebrow">MTech Thesis &nbsp;·&nbsp; FinRAGBench-V (EMNLP 2025)</div>
<h1 class="hero-title">SAGE</h1>
<div class="hero-subtitle">Structured Agentic Graph-Enhanced RAG with Self-Correction</div>
<div class="hero-lede">Multimodal agentic RAG for business intelligence: a controlled comparison of
vector retrieval, GraphRAG and HippoRAG against a fused, reranked and self-correcting pipeline
on real financial filings.</div>
<div class="hero-author"><b>Saptarshi Das</b> MTech Computer Science &amp; Data Processing, IIT Kharagpur</div>
<div class="stat-row">
  <div class="stat"><div class="stat-value">18</div><div class="stat-label">Documents</div></div>
  <div class="stat"><div class="stat-value">232</div><div class="stat-label">Questions</div></div>
  <div class="stat"><div class="stat-value">928</div><div class="stat-label">Evaluation runs</div></div>
  <div class="stat"><div class="stat-value">4</div><div class="stat-label">Configurations</div></div>
</div>
""", unsafe_allow_html=True)

tab_demo, tab_results, tab_category, tab_about = st.tabs([
    "Live Demo",
    "Benchmark Results",
    "Category Analysis",
    "About"
])


with tab_demo:
    kicker("Interactive")
    st.markdown("### Ask the corpus")
    st.caption("Choose the configurations to compare, pose a question, and read the answers side by side.")

    selected = st.pills(
        "Configurations",
        list(CONFIG_LABELS.values()),
        selection_mode="multi",
        default=list(CONFIG_LABELS.values()),
    ) or []

    SAMPLE_QUESTIONS = [
        "According to Howden Joinery Group Plc Annual Reports 2023, which is higher between cash at bank in 2022 and 2023?",
        "What regulations is Ambac subject to in the UK?",
        "Which companies compete with Ambac in the insurance sector?",
        "What sectors does Howden operate in?",
        "According to the SRCE 2022 Form 10-K, which year had a lower net income?",
        "What methods did the auditors use to ensure effectiveness of internal controls for Ambac in 2021?",
        "According to Craneware plc 2022 annual report, which region contributed the highest revenue?",
    ]

    sample = st.selectbox(
        "Sample questions",
        SAMPLE_QUESTIONS,
        index=None,
        placeholder="Pick a sample question, or write your own below",
    )
    query = st.text_area(
        "Your question",
        value=sample or "",
        height=90,
        placeholder="e.g. Which region contributed the highest revenue for Craneware in 2022?"
    )

    run_button = st.button("Run query", type="primary", disabled=not query.strip())

    if run_button and query.strip():
        selected = [c for c in CONFIG_LABELS.values() if c in selected]
        if not selected:
            st.warning("Select at least one configuration.")
        else:
            with st.spinner("Loading models (the first run can take about 30 seconds)…"):
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
                with col, st.container(border=True):
                    st.markdown(
                        f'<div class="card-title">{dot(config_name)}{config_name}</div>',
                        unsafe_allow_html=True,
                    )
                    with st.spinner("Running…"):
                        start = time.time()
                        try:
                            result = run_fn(query)
                            latency = time.time() - start

                            answer = result.get("answer", "")
                            clean_answer = re.sub(r'\[UNVERIFIED\]\s*', '', answer).strip()

                            st.write(clean_answer)

                            meta = f"{latency:.2f}s"
                            if config_name == "SAGE":
                                meta += (f" · {result.get('corrections', 0)} self-corrections"
                                         f" · {result.get('flagged', 0)} flagged claims")
                            st.caption(meta)

                            with st.expander("Sources"):
                                for s in result.get("sources", []):
                                    st.markdown(
                                        f"- {s.get('type','text').title()} · "
                                        f"`{s.get('doc_id','')}` p.{s.get('page','')}"
                                    )

                            if config_name == "GraphRAG":
                                paths = result.get("graph_paths", [])
                                if paths:
                                    with st.expander("Graph paths"):
                                        for p in paths[:8]:
                                            st.markdown(f"- `{p}`")

                            if config_name == "HippoRAG":
                                nodes = result.get("ranked_nodes", [])
                                if nodes:
                                    with st.expander("Ranked entities"):
                                        for name, ntype, score in nodes[:5]:
                                            st.markdown(f"- **{name}** ({ntype}) {score:.3f}")

                        except Exception as e:
                            st.error(f"Error: {str(e)[:200]}")


with tab_results:
    kicker("Benchmark")
    st.markdown("### Overall results")
    st.caption("232 questions × 4 configurations = 928 evaluation runs, scored automatically against the qrels.")

    try:
        results = load_results()
        summary = compute_summary(results)

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

        winners = []
        rows = []
        for metric_name, key in metrics_display:
            row = {"Metric": metric_name}
            vals = {c: summary.get(c, {}).get(key) for c in CONFIG_ORDER}
            for c in CONFIG_ORDER:
                row[CONFIG_LABELS[c]] = vals[c]
            rows.append(row)
            present = {c: v for c, v in vals.items() if v is not None}
            if present:
                pick = min if key == "latency" else max
                best = pick(present, key=present.get)
                winners.append((metric_name, CONFIG_LABELS[best], present[best], key))

        df = pd.DataFrame(rows).set_index("Metric")

        def highlight_best(row):
            vals = row.dropna()
            if vals.empty:
                return [""] * len(row)
            target = vals.min() if row.name.startswith("Avg Latency") else vals.max()
            return [
                "font-weight: 600; background-color: #e1e9da; color: #1c1f26"
                if v == target else "" for v in row
            ]

        styled = df.style.apply(highlight_best, axis=1).format("{:.3f}", na_rep="—")
        st.dataframe(styled, width="stretch", column_config={"_index": st.column_config.Column(width="medium")})
        st.caption("Highlighted cells mark the best configuration for each metric (lowest for latency).")

        st.markdown("")
        kicker("Leaders")
        st.markdown("### Best configuration by metric")
        cards = []
        for metric_name, label, val, key in winners:
            shown = f"{val:.2f}s" if key == "latency" else f"{val:.3f}"
            cards.append(
                f'<div class="winner"><div class="winner-metric">{metric_name}</div>'
                f'<div class="winner-name">{dot(label)}{label}</div>'
                f'<div class="winner-value">{shown}</div></div>'
            )
        st.markdown(f'<div class="winner-grid">{"".join(cards)}</div>', unsafe_allow_html=True)

    except FileNotFoundError:
        st.warning("eval/results.json not found. Run evaluation first.")


with tab_category:
    kicker("Breakdown")
    st.markdown("### Performance by question category")

    try:
        results = load_results()

        metric_key_map = {
            "Precision@5":                  "precision@5",
            "Faithfulness":                 "faithfulness",
            "F1 Score (short answer only)": "f1",
            "Answer Relevancy":             "answer_relevancy",
            "MRR":                          "mrr",
        }
        metric_choice = st.segmented_control(
            "Metric", list(metric_key_map), default="Precision@5"
        ) or "Precision@5"
        key = metric_key_map[metric_choice]

        by_cat = defaultdict(lambda: defaultdict(list))
        for r in results:
            val = r.get(key)
            if val is not None:
                by_cat[r["category"]][r["config"]].append(val)

        rows = []
        for cat in sorted(by_cat.keys()):
            row = {"Category": cat}
            for config in CONFIG_ORDER:
                vals = by_cat[cat].get(config, [])
                row[CONFIG_LABELS[config]] = round(statistics.mean(vals), 3) if vals else None
            rows.append(row)

        df = pd.DataFrame(rows).set_index("Category")
        order = [CONFIG_LABELS[c] for c in CONFIG_ORDER]
        long = df.reset_index().melt(id_vars="Category", var_name="Configuration", value_name="Score")

        chart = (
            alt.Chart(long)
            .mark_bar(cornerRadiusEnd=4, height={"band": 1})
            .encode(
                y=alt.Y("Category:N", sort=list(df.index), title=None,
                        scale=alt.Scale(paddingInner=0.3),
                        axis=alt.Axis(labelLimit=260, labelPadding=10, minExtent=210)),
                yOffset=alt.YOffset("Configuration:N", sort=order, scale=alt.Scale(paddingInner=0.15)),
                x=alt.X("Score:Q", title=metric_choice, axis=alt.Axis(tickCount=6, format=".2f")),
                color=alt.Color(
                    "Configuration:N",
                    scale=alt.Scale(domain=order, range=[CONFIG_COLORS[o] for o in order]),
                    legend=alt.Legend(orient="top", title=None, symbolType="circle"),
                ),
                tooltip=[
                    alt.Tooltip("Category:N"),
                    alt.Tooltip("Configuration:N"),
                    alt.Tooltip("Score:Q", format=".3f"),
                ],
            )
            .properties(height=560)
            .configure(background="transparent", font="Inter", autosize=alt.AutoSizeParams(type="fit", contains="padding"))
            .configure_view(stroke=None)
            .configure_axis(
                labelColor="#3d3f45", titleColor=INK_MUTED, labelFontSize=12, titleFontSize=12,
                titleFontWeight="normal", gridColor="#e1e6da", domain=False, ticks=False,
            )
            .configure_axisY(grid=False)
            .configure_legend(labelColor="#3d3f45", labelFontSize=12, symbolSize=90)
        )
        st.altair_chart(chart, use_container_width=True, theme=None)

        with st.expander("Show as table"):
            st.dataframe(df.style.format("{:.3f}", na_rep="—"), width="stretch")

        st.markdown("")
        kicker("Reference")
        st.markdown("### Category descriptions")
        st.markdown("""
| Category | Description | Expected winner |
|----------|-------------|-----------------|
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


with tab_about:
    kicker("Overview")
    st.markdown("### About the project")
    st.markdown("""
**SAGE (Structured Agentic Graph-Enhanced RAG with Self-Correction)** is a controlled benchmark
comparing four RAG architectures on real financial documents from FinRAGBench-V (EMNLP 2025).
""")

    kicker("Architecture")
    st.markdown("### Four configurations")

    architectures = [
        ("Vanilla RAG", "Config 1 · dense vector retrieval", """
- ChromaDB vector store with BAAI/bge-large-en-v1.5 embeddings (1024-dim)
- Smart metadata routing: table chunks for financial queries, figure chunks for chart queries
- Fastest config at 1.68s average latency
"""),
        ("GraphRAG", "Config 2 · knowledge-graph traversal", """
- Neo4j Knowledge Graph: 11,052 nodes, 23,077 edges
- Entity extraction via Groq LLM → 2-hop Cypher traversal with fuzzy matching
- Graph-discovered entities enrich ChromaDB query for better retrieval
- Best answer relevancy: 0.952
"""),
        ("HippoRAG", "Config 3 · personalized PageRank", """
- Same Neo4j KG as GraphRAG
- Personalized PageRank via Neo4j GDS plugin (damping=0.85, maxIterations=20)
- PageRank propagates relevance through entire graph — finds indirect connections
- Best retrieval: Precision@5 (0.103), Recall@5 (0.466), Faithfulness (0.745)
"""),
        ("SAGE", "Config 4 · fusion, reranking, self-correction", """
- Parallel retrieval: runs all 3 configs simultaneously
- Context fusion: deduplicates chunks by doc_id + page
- Reranking: BAAI/bge-reranker-large cross-encoder scores all fused chunks
- Self-correction: claim-level LLM judge (threshold 0.6), re-retrieval for borderline claims (0.3-0.6)
- Best ranking and factual accuracy: MRR (0.377), Exact Match (0.170), F1 (0.297)
"""),
    ]

    for pair in (architectures[:2], architectures[2:]):
        cols = st.columns(2)
        for col, (name, tag, body) in zip(cols, pair):
            with col, st.container(border=True):
                st.markdown(
                    f'<div class="card-title">{dot(name)}{name}</div>'
                    f'<div class="card-tag">{tag}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(body)

    st.markdown("")
    c1, c2, c3 = st.columns(3)
    with c1:
        kicker("Data")
        st.markdown("#### Dataset")
        st.markdown("""
- **FinRAGBench-V** (EMNLP 2025), a financial RAG benchmark with visual citation
- 51,219 English pages from 1,105 financial documents
- 539 human-annotated QA pairs across 7 question categories
- Subset used: 18 documents, 232 valid questions after principled document-first selection
""")
    with c2:
        kicker("Stack")
        st.markdown("#### Tech stack")
        st.markdown("""
- **Embeddings:** BAAI/bge-large-en-v1.5 (local, 1024-dim)
- **Reranker:** BAAI/bge-reranker-large (local, 560M parameters)
- **LLM:** GPT-4o Mini (generation + scoring), Groq Llama-3.3-70B (entity extraction)
- **Vector store:** ChromaDB (persistent, 6,181 chunks)
- **Graph DB:** Neo4j + GDS plugin (Docker, 11,052 nodes, 23,077 edges)
- **KG extraction:** GPT-4o Mini (5,256 chunks processed)
- **Vision:** GPT-4o (table + chart extraction, ~$1.50 total)
- **Framework:** Pure Python, no LangChain agent framework
""")
    with c3:
        kicker("Method")
        st.markdown("#### Evaluation")
        st.markdown("""
- **Corpus selection:** documents scored by (questions × modalities), selected before question filtering
- **Question filtering:** only questions where all gold pages are in the selected documents
- **Scoring:** fully automatic via qrels.tsv, no manual annotation
- **Short answers:** GPT-4o Mini post-processing for Table categories before EM/F1
- **Metrics:** Precision@5, Recall@5, MRR, Exact Match, F1, Faithfulness, Answer Relevancy, Latency
""")

st.markdown("""
<div class="footer">
  <span><b>Saptarshi Das</b> MTech Computer Science &amp; Data Processing, IIT Kharagpur</span>
  <span>FinRAGBench-V (EMNLP 2025)</span>
</div>
""", unsafe_allow_html=True)
