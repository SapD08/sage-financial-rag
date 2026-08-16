# Metrics Log

## Phase 2 — Preprocessing
- Documents: 18
- Valid questions: 232
- Text chunks: 5,256
- Table chunks: 909
- Figure chunks: 16
- Total chunks: 6,181
- Dropped: NASDAQ_AMBC_2022 (CID font encoding), 6 Financial Times PDFs (scanned images)
- Chunk size: 400 words, overlap: 50 words

## Phase 3A — ChromaDB Index
- Total chunks indexed: 6,181
- Embedding model: BAAI/bge-large-en-v1.5 (1024-dim)
- Metadata filter verified: type=table/figure/text
- BGE query prefix applied at retrieval time

## Phase 3B — Neo4j Knowledge Graph
- Nodes: 11,052
- Edges: 23,077
- Node types: Metric(4668), Person(2406), Company(1254), Regulation(1220), Sector(989), Entity(528)
- Edge types: REPORTS_METRIC(10134), SUBJECT_TO(8079), LED_BY(3348), OPERATES_IN(3178), COMPETES_WITH(1150)
- Extraction model: gpt-4o-mini
- Noise cleaned: 13 generic nodes removed

## Phase 4 — Retrievers
- Vanilla RAG: BGE Large embeddings + smart metadata filter (table/figure/text)
- GraphRAG: Entity extraction → 2-hop Cypher traversal → entity-enriched ChromaDB retrieval
- HippoRAG: Entity extraction → Personalized PageRank (GDS) → entity-enriched ChromaDB retrieval
- SAGE: Parallel fusion (vanilla + graph + hippo) → BGE reranker → draft answer → self-correction

## Phase 5 — SAGE Agent
- Architecture: Parallel retrieval + Context fusion + BGE reranker + Self-correction
- Reranker: BAAI/bge-reranker-large (2.1GB, domain-aligned with embedding model)
- Self-correction: Claim-level LLM judge (threshold 0.6), re-retrieval for borderline claims (0.3-0.6)
- Name: SAGE — Structured Agentic Graph-Enhanced RAG

## Phase 6 — Final Evaluation
- Dataset: FinRAGBench-V (EMNLP 2025)
- Documents: 18 | Questions: 232 | Configs: 4 | Total runs: 928
- Scoring: Automatic via qrels.tsv
- Short answer extraction: GPT-4o Mini post-processing for Table categories

### Overall Results

| Metric | Vanilla RAG | GraphRAG | HippoRAG | SAGE |
|--------|------------|----------|----------|------|
| Precision@5 | 0.085 | 0.101 | **0.103** | 0.099 |
| Recall@5 | 0.390 | 0.457 | **0.466** | 0.446 |
| MRR | 0.227 | 0.263 | 0.269 | **0.377** |
| Exact Match | 0.125 | 0.134 | 0.143 | **0.170** |
| F1 Score | 0.269 | 0.264 | 0.271 | **0.297** |
| Faithfulness | 0.671 | 0.728 | **0.745** | 0.702 |
| Answer Relevancy | 0.816 | **0.952** | 0.947 | 0.907 |
| Avg Latency | **1.68s** | 5.69s | 6.65s | 26.58s |

### Precision@5 by Category

| Category | Vanilla RAG | GraphRAG | HippoRAG | SAGE |
|----------|------------|----------|----------|------|
| Chart-Information Extraction | 0.109 | 0.109 | **0.145** | 0.091 |
| Chart-Time Sensitive | 0.000 | 0.067 | 0.067 | 0.067 |
| Table-Compare and Sort | 0.080 | 0.083 | **0.087** | 0.083 |
| Table-MultiPage | 0.100 | 0.100 | 0.100 | 0.075 |
| Table-Numerical Calculation | 0.108 | 0.115 | 0.115 | 0.112 |
| Text Inference | 0.070 | **0.095** | 0.092 | 0.095 |
| Text-MultiPage | 0.092 | 0.133 | **0.142** | **0.142** |

### F1 Score by Category (short answer only)

| Category | Vanilla RAG | GraphRAG | HippoRAG | SAGE |
|----------|------------|----------|----------|------|
| Table-Compare and Sort | 0.289 | 0.260 | 0.297 | **0.352** |
| Table-Numerical Calculation | 0.246 | **0.269** | 0.242 | 0.233 |

### Faithfulness by Category

| Category | Vanilla RAG | GraphRAG | HippoRAG | SAGE |
|----------|------------|----------|----------|------|
| Chart-Information Extraction | 0.536 | 0.618 | **0.745** | 0.355 |
| Chart-Time Sensitive | 0.667 | 0.967 | **1.000** | 0.800 |
| Table-Compare and Sort | 0.510 | 0.617 | 0.693 | 0.620 |
| Table-MultiPage | 0.637 | 0.688 | 0.688 | 0.637 |
| Table-Numerical Calculation | 0.585 | 0.583 | 0.544 | 0.512 |
| Text Inference | 0.839 | 0.882 | 0.874 | **0.918** |
| Text-MultiPage | 0.817 | 0.875 | **0.900** | 0.821 |

### Winners Summary
- Best Retrieval (Precision + Recall + Faithfulness): **HippoRAG**
- Best Ranking (MRR): **SAGE**
- Best Factual Accuracy (EM + F1): **SAGE**
- Best Answer Quality (Answer Relevancy): **GraphRAG**
- Fastest: **Vanilla RAG (1.68s)**
- Most Expensive/Slowest: **SAGE (26.58s)**