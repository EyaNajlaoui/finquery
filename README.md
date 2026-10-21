# FinQuery

**Ask questions about financial filings and get answers grounded in the source pages.**

FinQuery is a retrieval-augmented generation (RAG) system for 10-K, 10-Q, 8-K and earnings releases.
You load the filings, ask a question in plain English ("What was Boeing's gross margin in FY2022?"),
and FinQuery answers with citations to the exact document and page, computes ratios with a safe calculator
instead of letting the LLM do arithmetic, and says so when the documents do not contain the answer.

Every design choice is measured on a public benchmark (FinanceBench) with a fixed dev/test split.

> **Status: step 0 – foundations.** Repository structure, configuration, data pipeline, Docker setup,
> tests and CI are in place. See the [roadmap](#roadmap) for what comes next.

## Planned features

- **Ingestion** of PDF filings: text and tables (tables linearized row by row), page-level metadata
  (company, document type, period, page).
- **Hybrid retrieval**: dense embeddings (`bge-small-en-v1.5`) + BM25, fused with Reciprocal Rank Fusion,
  then reranked with a cross-encoder. Stored in **Qdrant** with metadata filters.
- **Grounded answers** with `[document, page]` citations and an explicit refusal when the evidence is missing.
- **Agent mode**: the LLM can search several times, filter by company/year and call a calculator tool
  for ratios and growth rates.
- **Evaluation**: Recall@k and MRR on evidence pages, numeric correctness checked by code, and an
  LLM judge for correctness and faithfulness, all reported on a held-out test set.
- **Streamlit interface**: chat, filters by company / document type / period, sources shown with each answer.

## Architecture

```mermaid
flowchart LR
    A[PDF filings] --> B[Parse text + tables]
    B --> C[Chunk + metadata]
    C --> D[(Qdrant)]
    C --> E[BM25 index]
    Q[Question] --> F[Hybrid search + RRF]
    D --> F
    E --> F
    F --> G[Cross-encoder rerank]
    G --> H[LLM - Groq]
    H --> I[Answer + citations]
    H <-.-> T[Calculator tool]
```

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Interface | Streamlit | fast to build, easy to demo |
| Vector store | Qdrant (local mode in dev, server in Docker) | payload filters, same API from laptop to production |
| Embeddings | `BAAI/bge-small-en-v1.5` | strong English retrieval, runs on CPU |
| Lexical search | `rank-bm25` | exact matches on tickers, line items and numbers |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | better ordering of the top candidates, CPU friendly |
| LLM | `openai/gpt-oss-120b` via Groq | tool calling, fast inference |
| PDF parsing | `pdfplumber`, `pypdf` | text and table extraction |
| Packaging | Docker, Docker Compose | one command to run app + database |
| Quality | pytest, ruff, GitHub Actions | tests and lint on every push |

All parameters (models, chunk size, top-k...) live in [`config/settings.yaml`](config/settings.yaml).

## Data

50 questions from [FinanceBench](https://github.com/patronus-ai/financebench) over 18 filings from
13 companies (1,242 pages), split into **20 dev** and **30 test** questions stratified by question type.
Details, selection table and file format: [`data/README.md`](data/README.md).

## Quickstart

Requirements: Python 3.10+ and a free [Groq API key](https://console.groq.com/keys).
Docker is optional.

### Local

```bash
git clone https://github.com/<your-username>/finquery.git
cd finquery
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .

cp .env.example .env             # Windows: copy .env.example .env   then add your GROQ_API_KEY
python scripts/download_data.py  # 18 PDFs (~17 MB) + dev/test files
streamlit run app/Home.py        # http://localhost:8501
```

### Docker

```bash
cp .env.example .env             # add your GROQ_API_KEY
docker compose up --build        # app on http://localhost:8501, Qdrant on http://localhost:6333
```

The `data/` folder is mounted into the container, so the downloaded PDFs and the index are shared
with the local setup. Downloaded models are cached in a Docker volume.

### Tests

```bash
pytest
ruff check .
```

## Project structure

```
finquery/
├── app/                    # Streamlit interface
│   ├── Home.py
│   └── pages/
├── config/settings.yaml    # all parameters
├── data/
│   ├── raw/                # PDFs (downloaded, not in git)
│   └── eval/               # dev.jsonl, test.jsonl
├── scripts/                # command-line entry points
│   └── download_data.py
├── src/finquery/
│   ├── config.py           # settings loading + env overrides
│   ├── data/               # FinanceBench loading, normalization, split
│   ├── ingestion/          # parsing, chunking (step 1)
│   ├── retrieval/          # dense, BM25, fusion, rerank (steps 2-3)
│   ├── generation/         # prompts, grounded answers (step 4)
│   ├── agent/              # tool-calling agent (step 5)
│   ├── tools/              # safe calculator
│   ├── sql/                # SQL safety checks (optional structured data)
│   └── evaluation/         # retrieval + answer metrics
├── tests/
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/ci.yml
```

## Roadmap

- [x] **Step 0 – Foundations**: structure, config, data download and split, Docker, tests, CI
- [ ] **Step 1 – Ingestion**: PDF parsing (text + tables), chunking, metadata
- [ ] **Step 2 – Indexing**: embeddings in Qdrant, BM25 index
- [ ] **Step 3 – Retrieval evaluation**: Recall@k / MRR on evidence pages, dense vs hybrid vs rerank
- [ ] **Step 4 – Generation**: grounded answers with citations, refusal, LLM judge
- [ ] **Step 5 – Agent**: multi-step search + calculator tool, traces
- [ ] **Step 6 – Streamlit app**: chat, filters, sources, upload of new filings
- [ ] **Step 7 – Production**: logging, caching, error handling, Docker image hardening
- [ ] **Step 8 – Final evaluation**: one run on the test set, results table in this README

## Results

Will be filled in at step 8 with the scores on the 30 test questions.

## Disclaimer

FinQuery is an educational and analysis tool. Its answers can be wrong and are **not investment advice**.

## License

Code: to be added. Evaluation data derived from FinanceBench is under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
