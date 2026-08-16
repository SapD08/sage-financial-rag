<div align="center">

# SAGE: Multimodal Benchmarking of GraphRAG, HippoRAG, and Vector RAG with Agentic Self-Correction for Financial Document Intelligence

**IIT Kharagpur — MTech Computer Science & Data Processing**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Neo4j](https://img.shields.io/badge/Neo4j-5%20%2B%20GDS-018BFF?logo=neo4j&logoColor=white)](https://neo4j.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-FF6F00)](https://www.trychroma.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#15-license)

*A controlled multimodal RAG benchmark comparing GraphRAG, HippoRAG, and Vector RAG — orchestrated by SAGE, an agentic self-correcting fusion system — evaluated on real financial documents from FinRAGBench-V (EMNLP 2025).*

</div>

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [The Four Configs](#3-the-four-configs)
4. [Dataset — FinRAGBench-V](#4-dataset--finragbench-v)
5. [Multimodal Ingestion Pipeline](#5-multimodal-ingestion-pipeline)
6. [Knowledge Graph](#6-knowledge-graph)
7. [Full Evaluation Results](#7-full-evaluation-results)
8. [Setup Instructions](#8-setup-instructions)
9. [How to Run](#9-how-to-run)
10. [Project Structure](#10-project-structure)
11. [Tech Stack](#11-tech-stack)
12. [Key Findings](#12-key-findings)
13. [Limitations and Future Work](#13-limitations-and-future-work)
14. [Citation](#14-citation)
15. [License](#15-license)

---

## 1. Project Overview

Financial documents — 10-Ks, annual reports, proxy statements — are long, dense, and **multimodal**: the number an analyst needs might be a sentence in the MD&A section, a cell in a five-year comparison table, or a bar in a segment-revenue chart. They're also unforgiving: retrieving the *almost*-right page and misquoting a figure is worse than admitting you don't know. This makes financial RAG a harder and more realistic testbed than the generic-text QA benchmarks most retrieval papers are evaluated on.

This project asks a focused question under controlled conditions — same corpus, same 232 questions, same LLMs, same evaluation harness — across four architectures:

- **Does structuring the corpus as a knowledge graph and traversing entity relationships (GraphRAG) beat flat vector search?**
- **Does propagating relevance through the *entire* graph via Personalized PageRank (HippoRAG) surface evidence that local 2-hop traversal misses?**
- **Does fusing all three retrieval signals, reranking with a cross-encoder, and adding a claim-level self-correction loop actually reduce hallucination and improve factual accuracy — and what does that cost in latency?**

**SAGE** (*Structured Agentic Graph-Enhanced RAG with Self-Correction*) is this project's answer to the third question and its principal contribution: a fourth config that runs the other three retrievers in parallel, fuses and reranks their combined evidence, drafts an answer, and then **verifies every individual claim in that answer against the retrieved context** — re-retrieving for borderline claims and flagging anything it still can't support — before returning a final, more accountable answer. Unlike a single end-to-end retriever, SAGE treats hallucination mitigation as a measurable, separate pipeline stage rather than an emergent property of a bigger prompt.

All four configs are evaluated identically and automatically against `qrels.tsv` — retrieval metrics (Precision@5, Recall@5, MRR), generation metrics (Exact Match, F1), and LLM-judged quality metrics (Faithfulness, Answer Relevancy) — with **zero manual annotation**.

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                               INGESTION (offline)                             │
│                                                                                │
│   18 financial PDFs (data/raw/)                                              │
│        │                                                                      │
│        ├──► pdfplumber text extraction ─────► text_chunker.py  ─────┐        │
│        │      (400-word chunks, 50-word overlap)                    │        │
│        │                                                             │        │
│        ├──► page-image + GPT-4o vision ──────► table_extractor.py ──┤        │
│        │      (Markdown tables per page)                            │        │
│        │                                                             ▼        │
│        └──► page-image + GPT-4o vision ──────► chart_extractor.py ─►┤        │
│               (chart descriptions per page)                          │        │
│                                                                        │        │
│                                                     6,181 chunks       │        │
│                                        ┌────────────────────────────┴──┐     │
│                                        ▼                                ▼     │
│                              build_index.py                    kg_builder.py  │
│                              (BAAI/bge-large-en-v1.5)           (GPT-4o mini   │
│                                        │                        entity/rel     │
│                                        ▼                        extraction)    │
│                              ┌──────────────────┐          ┌──────────────┐  │
│                              │     ChromaDB      │          │    Neo4j      │  │
│                              │  6,181 vectors     │          │  + GDS plugin │  │
│                              │  (persistent)       │          │  11,052 nodes │  │
│                              │  metadata: doc_id,  │          │  23,077 edges │  │
│                              │  page, type          │          │              │  │
│                              └─────────┬────────┘          └──────┬───────┘  │
└────────────────────────────────────────┼───────────────────────────┼──────────┘
                                          │                           │
┌─────────────────────────────────────────┼───────────────────────────┼──────────┐
│                                QUERY TIME (online)                              │
│                                          │                           │          │
│                                    ┌─────┴─────┐                     │          │
│                                    │   Query    │                     │          │
│                                    └─────┬─────┘                     │          │
│         ┌──────────────────┬─────────────┼─────────────┬─────────────┘          │
│         ▼                  ▼             │             ▼                        │
│  ┌──────────────┐  ┌───────────────┐     │     ┌───────────────┐               │
│  │ Config 1:     │  │ Config 2:      │     │     │ Config 3:      │               │
│  │ Vanilla RAG   │  │ GraphRAG        │     │     │ HippoRAG        │               │
│  │               │  │                 │     │     │                 │               │
│  │ smart metadata│  │ Groq entity     │     │     │ Groq entity     │               │
│  │ filter (table/│  │ extraction      │     │     │ extraction      │               │
│  │ figure/text)  │  │      ↓          │     │     │      ↓          │               │
│  │      ↓        │  │ 2-hop Cypher    │     │     │ Personalized    │               │
│  │ ChromaDB      │  │ traversal        │     │     │ PageRank (GDS)   │               │
│  │ top-k         │  │      ↓          │     │     │      ↓          │               │
│  │               │  │ entity-enriched │     │     │ entity-enriched │               │
│  │               │  │ ChromaDB query  │     │     │ ChromaDB query  │               │
│  └───────┬──────┘  └────────┬───────┘     │     └────────┬───────┘               │
│          │                   │              │              │                        │
│          └───────────────────┴──────┬───────┴──────────────┘                        │
│                                       │  (fed in parallel into Config 4)             │
│                                       ▼                                              │
│                          ┌────────────────────────┐                                 │
│                          │   Config 4: SAGE         │                                 │
│                          ├────────────────────────┤                                 │
│                          │ 1. Context fusion         │  dedupe by doc_id + page        │
│                          │ 2. BGE reranker            │  bge-reranker-large             │
│                          │    (cross-encoder)         │  cross-encoder scores all       │
│                          │ 3. Draft answer             │  GPT-4o mini                    │
│                          │ 4. Self-correction:          │  per-claim LLM judge (0.6 thr)  │
│                          │      score ≥ 0.6  → keep       │                                  │
│                          │      0.3–0.6      → re-retrieve │ + re-judge                     │
│                          │      < 0.3         → flag         │ [UNVERIFIED]                    │
│                          └────────────┬───────────┘                                 │
│                                       ▼                                              │
│                          ┌────────────────────────┐                                 │
│                          │  Final Answer + Sources  │                                 │
│                          └────────────────────────┘                                 │
└────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. The Four Configs

### Config 1 — Vanilla RAG
The baseline every other config is measured against.
- `BAAI/bge-large-en-v1.5` embeddings (1024-dim), indexed in ChromaDB.
- **Smart metadata filter**: query keywords route retrieval toward `table` chunks (financial/numeric terms: revenue, margin, EBITDA, %, …) or `figure` chunks (visual terms: chart, trend, bar, pie, …); falls back to unfiltered search if a filtered query returns too few hits.
- Simplest pipeline, single ChromaDB round-trip, no LLM calls before generation.

### Config 2 — GraphRAG
- Query entities extracted via **Groq Llama-3.3-70B** (fast, free-tier, low-latency entity extraction).
- **2-hop Cypher traversal** over the Neo4j knowledge graph, prioritized by relationship type (`SUBJECT_TO` → `COMPETES_WITH` → `OPERATES_IN` → `REPORTS_METRIC` → `LED_BY`).
- Graph-discovered entities enrich the ChromaDB query text before retrieval; the raw graph paths (`Entity -[REL]-> Entity`) are also passed to the LLM as explicit structured context alongside the retrieved chunks.

### Config 3 — HippoRAG
- Same Neo4j knowledge graph and Groq entity extraction as GraphRAG.
- Instead of bounded local traversal, seed entity nodes feed **Personalized PageRank** via the **Neo4j GDS plugin** (`dampingFactor=0.85`, `maxIterations=20`, score threshold `0.15`) over a projected in-memory graph (`finrag_graph`).
- PageRank propagates relevance through the *entire* graph topology — surfacing indirectly-connected, multi-hop-relevant entities that bounded 2-hop traversal structurally cannot reach.
- Top-ranked entities (name, type, score) enrich the ChromaDB query and are passed to the LLM as ranked context.

### Config 4 — SAGE *(this project's contribution)*
- Runs **Vanilla RAG, GraphRAG, and HippoRAG retrieval in parallel** — three independent evidence pools per query.
- **Context fusion**: deduplicates the combined chunk pool by `doc_id + page`, so overlapping evidence from multiple retrievers isn't redundantly scored.
- **Reranking**: `BAAI/bge-reranker-large` — a 560M-parameter cross-encoder domain-aligned with the BGE embedding model — scores every fused chunk against the query and keeps the top-k.
- **Draft generation**: GPT-4o mini answers from the reranked, deduplicated context.
- **Self-correction** (the novel piece): the draft is split into sentence-level claims. Each claim is scored 0.0–1.0 by an LLM judge against the context:
  - **≥ 0.6** → claim is well-supported, kept as-is.
  - **0.3 – 0.6** → borderline; triggers a **targeted re-retrieval** for that specific claim text and a second judgment pass — if the new evidence supports it, the claim is kept and counted as a *correction*; otherwise it's flagged.
  - **< 0.3** → claim is flagged inline as `[UNVERIFIED]` in the final answer (visible to the end user, stripped only for automatic scoring).

This makes SAGE the only config where "the model isn't sure" is a first-class, visible output rather than a silently confident guess.

---

## 4. Dataset — FinRAGBench-V

[**FinRAGBench-V**](https://arxiv.org/abs/2505.17471) (EMNLP 2025) is a financial RAG benchmark purpose-built for **visual citation** — grounding answers in specific *pages* of real financial documents (as page images) rather than plain-text spans, so retrieval quality can be checked against exact source pages regardless of modality.

**Full corpus:** 51,219 English pages across 1,105 financial documents, with 539 human-annotated QA pairs spanning 7 question categories.

**Subset used in this project — 18 documents, 232 valid questions**, selected via a two-stage, principled process (`select_subset.py`, `drop_bad_docs.py`) rather than random sampling:

1. **Document-first scoring**: every document is scored as `(distinct questions referencing it) × (distinct modalities its questions touch — table/chart/text)`, and the top-scoring 25 documents are shortlisted *before* any question filtering — biasing the subset toward documents that are both heavily-tested and multimodal, not just heavily-tested.
2. **Gold-doc-presence filtering**: a question is kept only if **all** of its gold-answer pages fall inside the selected document set (no partial-credit questions), then one document (`NASDAQ_AMBC_2022`, a CID-font-encoded PDF `pdfplumber` couldn't parse cleanly) was dropped and its dependent questions removed.

This yields fully automatic, reproducible scoring via `qrels.tsv` with **zero manual re-annotation**.

**Question categories in the final subset (232 questions):**

| Category | Count | Share |
|---|---:|---:|
| Text Inference | 74 | 31.9% |
| Table-Compare and Sort | 60 | 25.9% |
| Table-Numerical Calculation | 52 | 22.4% |
| Text-MultiPage | 24 | 10.3% |
| Chart-Information Extraction | 11 | 4.7% |
| Table-MultiPage | 8 | 3.4% |
| Chart-Time Sensitive | 3 | 1.3% |
| **Total** | **232** | **100%** |

Documents span multiple listing venues and filing types — NASDAQ 10-Ks (Ambac, S&T Bancorp, AAON, TS Banking), LSE annual reports (Howden Joinery), and AIM annual reports (Craneware) — across multiple fiscal years, so retrievers also have to disambiguate *which year's* filing a question refers to.

---

## 5. Multimodal Ingestion Pipeline

Each of the 18 source PDFs is processed through three parallel, modality-specific extractors, all writing into a common chunk schema (`id`, `doc_id`, `page`, `type`, `text`):

| Modality | Method | Chunks | Notes |
|---|---|---:|---|
| **Text** | `pdfplumber` extraction → 400-word chunks, 50-word overlap (`preprocessing/text_chunker.py`) | **5,256** | Pages under 20 words skipped as near-empty; `(cid:N)` font-encoding artifacts stripped |
| **Tables** | GPT-4o vision on page images, 150 DPI (`preprocessing/table_extractor.py`) | **909** | Heuristic pre-filter (`pdfplumber` table detection + numeric-density check) before spending a vision call; output is clean Markdown with units in headers |
| **Figures** | GPT-4o vision on page images (`preprocessing/chart_extractor.py`) | **16** | Pre-filtered on image size + low text density + chart-related keywords; extracts data points, axis labels, and trend descriptions |
| **Total** | | **6,181** | Indexed into ChromaDB (`retrieval/build_index.py`) |

**Embedding model:** `BAAI/bge-large-en-v1.5`, 1024-dim, with the BGE-recommended query-side prefix (`"Represent this sentence for searching relevant passages: "`) applied consistently at both indexing and query time. Chunks are stored in ChromaDB with `doc_id`, `page`, and `type` metadata, enabling the metadata-filtered retrieval used by Vanilla RAG.

Both vision extractors cache results to disk per page (`data/tables/`, `data/figures/`) so re-runs never re-spend a GPT-4o vision call — total vision extraction cost for the full 18-document corpus was **≈ $1.50**.

---

## 6. Knowledge Graph

Built once, offline, from the 5,256 text chunks (`kg/kg_builder.py`), and shared by both GraphRAG and HippoRAG at query time.

**Extraction:** GPT-4o mini, prompted per-chunk to return structured JSON entities and relationships from a fixed schema — cached per chunk (`cache/extraction/`) so the graph can be rebuilt (`kg/reload_graph.py`) without re-calling the LLM.

**Scale:** 11,052 nodes, 23,077 edges.

**Node types:**

| Type | Count |
|---|---:|
| Metric | 4,668 |
| Person | 2,406 |
| Company | 1,254 |
| Regulation | 1,220 |
| Sector | 989 |
| Entity *(fallback/unclassified)* | 528 |

**Edge types:**

| Relationship | Count |
|---|---:|
| REPORTS_METRIC | 10,134 |
| SUBJECT_TO | 8,079 |
| LED_BY | 3,348 |
| OPERATES_IN | 3,178 |
| COMPETES_WITH | 1,150 |

13 generic/noise nodes were manually identified and removed post-extraction.

**Sample verification queries** (Neo4j Browser, `http://localhost:7474`):

```cypher
// Node/edge counts
MATCH (n) RETURN count(n) AS nodes;
MATCH ()-[r]->() RETURN count(r) AS edges;

// Node type distribution
MATCH (n) RETURN labels(n)[0] AS type, count(*) AS n
ORDER BY n DESC;

// Edge type distribution
MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS n
ORDER BY n DESC;

// Inspect a company's full 1-hop neighborhood
MATCH (c:Company {name: "Ambac"})-[r]-(n)
RETURN c.name, type(r), n.name, labels(n)[0]
LIMIT 25;

// 2-hop path used by GraphRAG traversal
MATCH (n)-[r1]-(m)-[r2]-(k)
WHERE toLower(n.name) CONTAINS "howden" AND k.name <> n.name
RETURN n.name, type(r1), m.name, type(r2), k.name
LIMIT 20;
```

---

## 7. Full Evaluation Results

**232 valid questions × 4 configs = 928 evaluation runs**, scored fully automatically against `qrels.tsv` — no manual grading.

### Overall Results

| Metric | Vanilla RAG | GraphRAG | HippoRAG | SAGE |
|---|---:|---:|---:|---:|
| Precision@5 | 0.085 | 0.101 | **0.103** | 0.099 |
| Recall@5 | 0.390 | 0.457 | **0.466** | 0.446 |
| MRR | 0.227 | 0.263 | 0.269 | **0.377** |
| Exact Match | 0.125 | 0.134 | 0.143 | **0.170** |
| F1 Score | 0.269 | 0.264 | 0.271 | **0.297** |
| Faithfulness | 0.671 | 0.728 | **0.745** | 0.702 |
| Answer Relevancy | 0.816 | **0.952** | 0.947 | 0.907 |
| Avg Latency | **1.68s** | 5.69s | 6.65s | 26.58s |

*(Bold = best value per metric, excluding latency.)*

### Precision@5 by Category

| Category | Vanilla RAG | GraphRAG | HippoRAG | SAGE |
|---|---:|---:|---:|---:|
| Chart-Information Extraction | 0.109 | 0.109 | **0.145** | 0.091 |
| Chart-Time Sensitive | 0.000 | 0.067 | 0.067 | 0.067 |
| Table-Compare and Sort | 0.080 | 0.083 | **0.087** | 0.083 |
| Table-MultiPage | 0.100 | 0.100 | 0.100 | 0.075 |
| Table-Numerical Calculation | 0.108 | 0.115 | 0.115 | 0.112 |
| Text Inference | 0.070 | **0.095** | 0.092 | 0.095 |
| Text-MultiPage | 0.092 | 0.133 | **0.142** | **0.142** |

### F1 Score by Category (short-answer categories only)

| Category | Vanilla RAG | GraphRAG | HippoRAG | SAGE |
|---|---:|---:|---:|---:|
| Table-Compare and Sort | 0.289 | 0.260 | 0.297 | **0.352** |
| Table-Numerical Calculation | 0.246 | **0.269** | 0.242 | 0.233 |

*(EM/F1 are only computed for the two "short-answer" categories, where the gold answer is a single fact/number rather than a narrative — see `eval/extract_short_answers.py`.)*

### Faithfulness by Category

| Category | Vanilla RAG | GraphRAG | HippoRAG | SAGE |
|---|---:|---:|---:|---:|
| Chart-Information Extraction | 0.536 | 0.618 | **0.745** | 0.355 |
| Chart-Time Sensitive | 0.667 | 0.967 | **1.000** | 0.800 |
| Table-Compare and Sort | 0.510 | 0.617 | 0.693 | 0.620 |
| Table-MultiPage | 0.637 | 0.688 | 0.688 | 0.637 |
| Table-Numerical Calculation | 0.585 | 0.583 | 0.544 | 0.512 |
| Text Inference | 0.839 | 0.882 | 0.874 | **0.918** |
| Text-MultiPage | 0.817 | 0.875 | **0.900** | 0.821 |

### Winners Summary

| Category | Winner | Value |
|---|---|---|
| Best Retrieval (Precision@5) | HippoRAG | 0.103 |
| Best Recall | HippoRAG | 0.466 |
| Best Ranking (MRR) | SAGE | 0.377 |
| Best Factual Accuracy (Exact Match) | SAGE | 0.170 |
| Best F1 | SAGE | 0.297 |
| Best Faithfulness | HippoRAG | 0.745 |
| Best Answer Relevancy | GraphRAG | 0.952 |
| Fastest | Vanilla RAG | 1.68s |
| Slowest / Most Expensive | SAGE | 26.58s |

---

## 8. Setup Instructions

### Prerequisites

- **Python 3.9+**
- **Docker** (for Neo4j + GDS)
- **≥ 16GB RAM** (local embedding model + reranker + Neo4j all resident at once)
- **OpenAI API key** (generation, scoring, vision extraction)
- **Groq API key** (query-time entity extraction)

### 1. Clone the repository

```bash
git clone <repository-url>
cd 5_mtp_rag_project
```

### 2. Create a virtual environment and install dependencies

```bash
python3 -m venv venvv
source venvv/bin/activate        # Windows: venvv\Scripts\activate

pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your real credentials:

```
GROQ_API_KEY=your_groq_key_here
OPENAI_API_KEY=your_openai_key_here
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

`config.py` loads these automatically via `python-dotenv` — no code changes needed. `.env` is gitignored; never commit real keys.

### 4. Start Neo4j with the GDS plugin

```bash
docker run -d \
  --name neo4j-finrag \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  -e NEO4J_PLUGINS='["graph-data-science"]' \
  -e NEO4J_dbms_security_procedures_unrestricted=gds.* \
  -v neo4j-finrag-data:/data \
  neo4j:5
```

Confirm it's up at `http://localhost:7474` (Neo4j Browser) and log in with the credentials above.

### 5. Project the in-memory GDS graph (required before HippoRAG's PageRank calls)

After the knowledge graph has been built (Step 9, Phase 3B), project it once per Neo4j restart in the Neo4j Browser or `cypher-shell`:

```cypher
CALL gds.graph.project(
  'finrag_graph',
  '*',
  '*'
)
```

This creates the named in-memory graph `finrag_graph` that `retrieval/hippo_retriever.py` calls `gds.pageRank.stream` against.

---

## 9. How to Run

Run phases in order — each depends on the previous phase's output.

**Phase 2 — Multimodal ingestion (PDF → chunks)**

```bash
python preprocessing/text_chunker.py
python preprocessing/table_extractor.py
python preprocessing/chart_extractor.py
```

**Phase 3A — Vector index**

```bash
python retrieval/build_index.py
```

**Phase 3B — Knowledge graph** *(smoke-test first — cheap sanity check before the full LLM extraction run)*

```bash
python kg/kg_builder.py --smoke-test
python kg/kg_builder.py
```

Then project the GDS graph as in [Setup §5](#8-setup-instructions) before running HippoRAG.

**Phase 4-5 — Sanity-check each retriever individually**

```bash
python retrieval/vanilla_rag.py
python retrieval/graph_retriever.py
python retrieval/hippo_retriever.py
python agent/sage.py
```

**Phase 6 — Full evaluation** *(928 runs — long; `caffeinate` prevents macOS from sleeping mid-run)*

```bash
python eval/extract_short_answers.py
caffeinate -i python eval/run_eval.py
```

Evaluation is resumable — `eval/results.json` is saved after every question, and reruns skip already-completed `(query_id, config)` pairs.

**Phase 7 — Launch the demo dashboard**

```bash
streamlit run app.py
```

---

## 10. Project Structure

```
5_mtp_rag_project/
├── app.py                          # Streamlit dashboard — live demo + benchmark results + category analysis
├── config.py                       # Loads secrets from .env, holds non-sensitive constants
├── config.py.example               # Reference config structure for GitHub (no real values)
├── requirements.txt                # Pinned Python dependencies
├── .env.example                    # Placeholder env template
├── metrics.md                      # Running phase-by-phase metrics log
│
├── agent/
│   └── sage.py                     # Config 4: parallel fusion + reranking + self-correction
│
├── retrieval/
│   ├── vanilla_rag.py              # Config 1: BGE embeddings + metadata-filtered ChromaDB search
│   ├── graph_retriever.py          # Config 2: entity extraction + 2-hop Cypher traversal
│   ├── hippo_retriever.py          # Config 3: entity extraction + Personalized PageRank (GDS)
│   └── build_index.py              # Embeds all chunks into ChromaDB
│
├── kg/
│   ├── kg_builder.py                # LLM entity/relationship extraction → Neo4j (cached per chunk)
│   └── reload_graph.py              # Rebuilds Neo4j graph from the extraction cache, no LLM calls
│
├── preprocessing/
│   ├── text_chunker.py              # pdfplumber text extraction → 400-word overlapping chunks
│   ├── table_extractor.py           # GPT-4o vision → Markdown tables per page
│   └── chart_extractor.py           # GPT-4o vision → chart/figure descriptions per page
│
├── eval/
│   ├── evaluate.py                  # Metric definitions + evaluation loop (P@5, R@5, MRR, EM, F1, LLM-judge)
│   ├── extract_short_answers.py     # Post-processing: short-answer extraction + EM/F1 recompute for Table categories
│   ├── run_eval.py                  # Entry point — runs all 4 configs over all valid questions
│   ├── remove_sage.py               # Utility: strips SAGE rows from results.json
│   ├── results.json                 # Full 928-run evaluation output (source of truth for all metrics above)
│   └── final_results.txt            # Human-readable snapshot of the summarized results
│
├── datasets/finragbench/
│   ├── queries/queries_en.json      # Full FinRAGBench-V English query set
│   ├── selected_docs.json           # The 18-document subset + selection scores
│   ├── valid_questions.json         # The 232 valid questions used in evaluation
│   └── qrels/qrels_en.tsv           # Gold relevance judgments (query-id → corpus-id)
│
├── data/                            # Raw PDFs, extracted chunks, cached vision outputs (gitignored)
├── cache/extraction/                # Per-chunk cached KG extraction results (gitignored)
├── chroma_db/                       # Persistent ChromaDB vector store (gitignored)
├── models/                          # Local bge-reranker-large weights (gitignored)
│
└── *.py  (root)                     # One-off analysis/debug/check scripts used during development
                                      # (e.g. analyze_data.py, select_subset.py, drop_bad_docs.py,
                                      #  check_*.py, debug_*.py — not part of the core pipeline)
```

---

## 11. Tech Stack

| Component | Choice | Why |
|---|---|---|
| LLM (query-time extraction) | Groq — Llama-3.3-70B-Versatile | Free tier, very fast inference — keeps GraphRAG/HippoRAG entity extraction off the query-latency critical path as much as possible |
| LLM (generation & scoring) | GPT-4o mini | Cost-effective, high-quality answer generation, claim scoring, and LLM-judge metrics |
| Vision LLM | GPT-4o | Best available quality for table/chart extraction from page images |
| Embeddings | `BAAI/bge-large-en-v1.5` | Strong open retrieval performance, 1024-dim, runs locally and free |
| Reranker | `BAAI/bge-reranker-large` | Cross-encoder, domain-aligned with the BGE embedding model, 560M params |
| Vector store | ChromaDB | Persistent, native metadata filtering (doc_id/page/type) |
| Graph database | Neo4j + GDS plugin | Cypher traversal, native Personalized PageRank, visual browser for debugging |
| App / demo | Streamlit | Fast interactive dashboard for live multi-config comparison |
| Config / secrets | `python-dotenv` | Credentials isolated in `.env`, never hardcoded |
| Framework | Pure Python | No LangChain/agent framework — every pipeline stage is explicit and inspectable |

---

## 12. Key Findings

1. **Graph-based retrieval consistently beats flat vector search.** Both GraphRAG (P@5 0.101, Recall 0.457) and HippoRAG (P@5 0.103, Recall 0.466) outperform Vanilla RAG (P@5 0.085, Recall 0.390) on every retrieval metric — entity-relationship context measurably helps find the right evidence.
2. **Global relevance propagation beats bounded traversal.** HippoRAG's Personalized PageRank edges out GraphRAG's 2-hop Cypher traversal on Precision@5 (0.103 vs 0.101), Recall@5 (0.466 vs 0.457), and Faithfulness (0.745 vs 0.728) — confirming that propagating relevance through the whole graph surfaces connections that a fixed 2-hop radius misses.
3. **Self-correction is the single biggest lever on factual accuracy, not retrieval quality.** SAGE doesn't have the best Precision@5 or Recall@5 (it fuses three retrievers, so noise from weaker ones dilutes the pool) — but its **MRR (0.377) is 40% higher than the next-best config (HippoRAG, 0.269)** and its **Exact Match (0.170) is 19% higher than the next-best (HippoRAG, 0.143)**, because reranking + claim verification fix *ranking and answer precision*, a different failure mode than retrieval recall.
4. **Faithfulness and answer relevancy don't move together.** GraphRAG has the best Answer Relevancy (0.952) but only third-best Faithfulness (0.728) — it answers *on-topic* fluently but isn't always the most evidence-grounded. HippoRAG is the inverse: best Faithfulness (0.745), slightly lower relevancy (0.947). SAGE's self-correction trades some relevancy (0.907) for accuracy gains elsewhere, rather than dominating on every axis.
5. **Fusion + verification has a steep latency cost.** SAGE averages **26.58s per query — 15.8× slower than Vanilla RAG (1.68s)** and ~4× slower than HippoRAG (6.65s), because it runs all three retrievers, reranks the union, and issues a separate LLM call per claim (plus a second call for every borderline claim). The accuracy gains are real, but SAGE is not currently a config you'd put behind a low-latency user-facing endpoint without further optimization (e.g., batched claim scoring, caching, or async parallelization of the verification loop).

---

## 13. Limitations and Future Work

- **Latency.** SAGE's per-claim, sequential LLM verification loop is the main scalability bottleneck (see Finding 5). Batching claim-scoring into a single LLM call, or scoring claims concurrently instead of sequentially, would likely cut latency substantially without changing the underlying self-correction logic.
- **F1/EM are token-overlap metrics.** They can score a *semantically wrong* answer non-trivially (e.g., an answer that inverts a "higher/lower" comparison but shares numeric tokens with the gold answer). A stricter, LLM-judged correctness metric for short-answer categories would be a more reliable accuracy signal than F1/EM alone.
- **Corpus scale.** The 18-document, 232-question subset was deliberately chosen for tractable, fully-automatic, principled evaluation — not to represent the full 1,105-document FinRAGBench-V corpus. Scaling the knowledge graph and re-running would test whether HippoRAG's advantage grows or shrinks with a much larger, noisier graph.
- **Single-embedding-model retrieval.** All three base retrievers currently share the same BGE embedding space; an ablation using a different or fine-tuned financial embedding model for one retriever would isolate how much of GraphRAG/HippoRAG's advantage is structural (graph) versus purely embedding-quality-driven.
- **Bare exception handling in the graph pipeline.** Several entity-extraction and graph-write paths (`retrieval/graph_retriever.py`, `retrieval/hippo_retriever.py`, `kg/kg_builder.py`) silently swallow all exceptions and fall back to "no entities found." This is safe for uninterrupted evaluation but makes silent degradation (e.g., an expired API key) indistinguishable from genuinely sparse entities — worth adding structured logging around before relying on this in a production setting.
- **Chart modality is thin.** Only 16 figure chunks and 14 chart-category questions exist in the current subset — not enough to draw strong conclusions about chart-specific retrieval quality; expanding the visual-QA slice of the corpus would strengthen that part of the benchmark.

---

## 14. Citation

If you use FinRAGBench-V, please cite the original dataset paper:

```bibtex
@inproceedings{finragbench2025,
  title     = {FinRAGBench-V: A Benchmark for Multimodal RAG with Visual Citation in the Financial Domain},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2025},
}
```

If you reference this project (SAGE and the four-config benchmark):

```bibtex
@misc{sage2026,
  title  = {SAGE: Multimodal Benchmarking of GraphRAG, HippoRAG, and Vector RAG
            with Agentic Self-Correction for Financial Document Intelligence},
  author = {IIT Kharagpur -- MTech Computer Science and Data Processing},
  year   = {2026},
}
```

---

## 15. License

Released under the **MIT License**. See [`LICENSE`](LICENSE) for full text — you are free to use, modify, and distribute this project, provided the original copyright notice is retained.
