# Data

FinQuery is evaluated on a subset of **[FinanceBench](https://github.com/patronus-ai/financebench)**
(Islam et al., 2023, *FinanceBench: A New Benchmark for Financial Question Answering*), a set of
questions written by financial analysts about real SEC filings and earnings releases, each with a
reference answer and the evidence page in the source PDF.

FinanceBench is released under the **CC BY-NC 4.0** license (non-commercial use, attribution required).
The files in `eval/` are derived from it and keep the same license.

## Contents

| Path | In git | Content |
|---|---|---|
| `raw/*.pdf` | no | 18 source filings (≈17 MB), downloaded by `scripts/download_data.py` |
| `eval/dev.jsonl` | yes | 20 questions used while building and tuning the system |
| `eval/test.jsonl` | yes | 30 questions kept aside, used **once** for the final evaluation |
| `documents.jsonl` | yes | company, document type and fiscal year of each document |
| `processed/pages/` | no | parsed text of each page, one file per document (cache) |
| `processed/chunks.jsonl` | no | chunks with metadata, built by `scripts/ingest.py` |
| `qdrant/` | no | local vector index (created in step 2) |

Rebuild everything with:

```bash
python scripts/download_data.py
```

The script is idempotent: PDFs already present are skipped, and the split is reproducible (seed 42).

## Selection

The open-source FinanceBench sample has 150 questions over 84 documents. To fit a laptop, FinQuery keeps
**18 documents** chosen so that the questions about them add up to **50 questions**, with no question
written by us. The selection covers 13 companies and 4 document types (10-K, 10-Q, 8-K, earnings release),
1,242 pages in total.

| Document | Company | Type | Year | Pages | Dev | Test |
|---|---|---|---|---:|---:|---:|
| AMD_2022_10K | AMD | 10-K | 2022 | 121 | 4 | 3 |
| AMCOR_2023_10K | Amcor | 10-K | 2023 | 156 | 2 | 2 |
| BOEING_2022_10K | Boeing | 10-K | 2022 | 190 | 1 | 6 |
| AMAZON_2017_10K | Amazon | 10-K | 2017 | 85 | 0 | 2 |
| AMAZON_2019_10K | Amazon | 10-K | 2019 | 83 | 1 | 0 |
| BESTBUY_2023_10K | Best Buy | 10-K | 2023 | 75 | 0 | 3 |
| COSTCO_2021_10K | Costco | 10-K | 2021 | 76 | 1 | 0 |
| NETFLIX_2015_10K | Netflix | 10-K | 2015 | 72 | 1 | 0 |
| NETFLIX_2017_10K | Netflix | 10-K | 2017 | 73 | 0 | 1 |
| NIKE_2019_10K | Nike | 10-K | 2019 | 104 | 0 | 1 |
| BESTBUY_2024Q2_10Q | Best Buy | 10-Q | 2024 | 30 | 2 | 1 |
| Pfizer_2023Q2_10Q | Pfizer | 10-Q | 2023 | 72 | 0 | 3 |
| JOHNSON_JOHNSON_2023_8K_dated-2023-08-30 | Johnson & Johnson | 8-K | 2023 | 27 | 1 | 2 |
| AMCOR_2023Q4_EARNINGS | Amcor | Earnings | 2023 | 14 | 1 | 1 |
| JOHNSON_JOHNSON_2022Q4_EARNINGS | Johnson & Johnson | Earnings | 2022 | 24 | 1 | 1 |
| MGMRESORTS_2022Q4_EARNINGS | MGM Resorts | Earnings | 2022 | 15 | 1 | 2 |
| PEPSICO_2023Q1_EARNINGS | PepsiCo | Earnings | 2023 | 16 | 1 | 1 |
| ULTABEAUTY_2023Q4_EARNINGS | Ulta Beauty | Earnings | 2023 | 9 | 3 | 1 |
| **Total** | | | | **1,242** | **20** | **30** |

## Dev / test split

The split is **stratified by question type** so both sets have the same mix:

| Question type | Dev | Test | What it tests |
|---|---:|---:|---|
| novel-generated | 11 | 16 | varied questions written by analysts |
| domain-relevant | 6 | 10 | questions an analyst would ask about any company |
| metrics-generated | 3 | 4 | numbers to extract or compute (ratios, growth...) |

All prompt and parameter tuning is done on **dev**. The **test** set is only run at the end, so the
reported scores are not inflated by tuning on the same questions.

## Format of `eval/*.jsonl`

One question per line:

```json
{
  "financebench_id": "financebench_id_00995",
  "company": "AMD",
  "doc_name": "AMD_2022_10K",
  "doc_type": "10k",
  "doc_period": 2022,
  "question_type": "domain-relevant",
  "question_reasoning": "Information extraction",
  "question": "What are the major products and services that AMD sells as of FY22?",
  "answer": "AMD sells server microprocessors (CPUs) and graphics processing units (GPUs), ...",
  "justification": null,
  "evidence": [{"text": "Overview\nWe are a global semiconductor company ...", "doc_name": "AMD_2022_10K", "page": 4}],
  "evidence_pages": [4]
}
```

**Page numbers are 1-based** (page 1 = first page of the PDF). The original FinanceBench field
`evidence_page_num` is 0-based; `normalize()` in `src/finquery/data/financebench.py` adds 1, so pages
match what a PDF viewer shows and what the ingestion step stores in the chunk metadata.
