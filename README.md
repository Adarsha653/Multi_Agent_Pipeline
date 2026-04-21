# Multi-Agent Research Pipeline

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-1.1.6-purple)
![LangChain](https://img.shields.io/badge/LangChain-1.2.15-black)
![Groq](https://img.shields.io/badge/Groq-Llama_3.3_70B-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.135.3-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

An autonomous multi-agent system that researches any topic end-to-end: web search (with optional **PDF RAG** via Qdrant and local embeddings), synthesis, a structured markdown report, critic revisions, automatic scoring, and optional **cross-session memory** (JSON on disk). Built on a free-tier friendly stack (Groq, DuckDuckGo, open-source libraries). **pytest** runs in CI; optional **API keys** and **rate limits** protect research and report routes (see `.env.example`).

---

## Live Demo

Hugging Face Space: https://huggingface.co/spaces/AdarshaAryal653/multi-agent-pipeline

---

## Agents

| Agent | Role |
| --- | --- |
| Supervisor | Pure logic routing — decides which agent runs next (no LLM call) |
| Search Agent | Generates 3 diverse sub-queries (overview / recent / deeper angle) and up to 9 primary DDGS hits (3×3, deduped + trimmed); sparse sub-queries pull **alternate DDGS** phrasing and **Wikipedia**; optional **PDF RAG** chunks (Qdrant + local embeddings) are merged first when **`document_ids`** are supplied |
| Analysis Agent | Synthesizes raw results into structured insights |
| Writer Agent | Produces a formatted markdown report with revision support when the critic requests changes |
| Critic Agent | Reviews the report and returns APPROVED or REVISE |
| Evaluator | Scores the final report on relevance, completeness, clarity, structure, and overall (1–10) plus short feedback |

---

## Tech Stack

| Layer | Tool | Cost |
| --- | --- | --- |
| Agent orchestration | LangGraph | Free (OSS) |
| LLM framework | LangChain | Free (OSS) |
| LLM | Llama 3.3 70B via Groq | Free tier |
| Web search | DuckDuckGo (`ddgs`); **Wikipedia** opensearch when DDGS returns fewer than **`WEB_SEARCH_SPARSE_THRESHOLD`** hits | Free |
| Optional PDF RAG | **Qdrant** + **sentence-transformers** (local embeddings); see **API endpoints** and **`.env.example`** | Free (OSS); Qdrant is self-hosted or cloud |
| API | FastAPI + Uvicorn | Free (OSS) |

**Total cost to run:** $0 on free tiers (Groq API key required).

---

## Project structure

```
Multi_Agent_Pipeline/
├── agents/
│   ├── supervisor.py
│   ├── search_agent.py
│   ├── analysis_agent.py
│   ├── writer_agent.py
│   └── critic_agent.py
├── graph/
│   ├── state.py
│   └── pipeline.py
├── tools/
│   └── search_tools.py
├── api/
│   └── main.py
├── eval/
│   └── evaluator.py
├── utils/
│   ├── logger.py
│   ├── groq_llm.py
│   ├── rag_store.py
│   ├── research_memory.py
│   ├── report_outcome.py
│   └── api_auth.py
├── tests/
│   ├── test_supervisor.py
│   ├── test_search_tools.py
│   ├── test_api_auth.py
│   ├── test_report_routes_auth.py
│   ├── test_health_ready.py
│   ├── test_research_memory.py
│   ├── test_report_outcome.py
│   ├── test_evaluator_quota.py
│   └── test_rag_chunking.py
├── .github/
│   └── workflows/ci.yml
├── reports/          # optional local artifacts / examples
├── logs/             # session logs from PipelineLogger
├── ui.html
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## Setup

1. **Clone the repo**

```bash
git clone https://github.com/Adarsha653/Multi_Agent_Pipeline.git
cd Multi_Agent_Pipeline
```

2. **Create and activate a virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Environment variables**

Copy `.env.example` to `.env` and set at least:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Optional LangSmith tracing (see comments in `.env.example`):

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=multi-agent-pipeline
```

5. **Run the pipeline (CLI demo)**

```bash
python3 -m graph.pipeline
```

This invokes `run_pipeline` with a built-in sample query and prints the report and scores to the terminal.

6. **Run the API + web UI**

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000/` for `ui.html`. The Docker image uses `uvicorn` on port **7860** (Hugging Face Spaces convention).

---

## API endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/` | Serves the web UI (`ui.html`) |
| POST | `/research` | Runs the full pipeline; saves the markdown report under `/tmp/reports` |
| POST | `/research/stream` | Same pipeline as **`/research`**, but responds with **Server-Sent Events** (`text/event-stream`): **`type: step`** messages during the graph, then one **`type: complete`** JSON payload (same shape as **`/research`**) after the report is scored and saved |
| POST | `/documents/upload` | Multipart **PDF** upload → chunk, **sentence-transformers** embeddings, **Qdrant** upsert; returns **`document_id`**. Use the same auth as **`POST /research`** when **`PIPELINE_API_KEY`** is set. |
| GET | `/reports` | Lists saved reports **newest first** (by file mtime). Each entry: **`filename`**, **`generated_at`** (local `YYYY-MM-DD HH:MM:SS`). Files whose body contains a pipeline **failure stub** (`Report generation failed:` / `Analysis failed:`) are **omitted** (they are not re-saved on new runs; old ones are hidden from this list). |
| GET | `/reports/{filename}` | Returns report metadata and markdown content |
| GET | `/reports/{filename}/download` | Downloads the report as a file |
| GET | `/health` | Liveness — process up (no auth, not rate-limited) |
| GET | `/ready` | Readiness — **503** if `GROQ_API_KEY` missing (no Groq call) |

Optional API protection (see `.env.example`): **`PIPELINE_API_KEY`** for **`POST /research`**, **`POST /research/stream`**, and **`POST /documents/upload`**; **`PIPELINE_REPORTS_API_KEY`** or the same pipeline key for **`GET /reports*`**. Rate limits use **`slowapi`** (configurable via env). The bundled **`ui.html`** does not send API keys—leave keys unset for a public browser demo, or use **`curl`** / a custom client with **`X-API-Key`** when keys are enabled.

**PDF RAG (optional):** Start **Qdrant** (for example `docker run -p 6333:6333 qdrant/qdrant`), set **`QDRANT_URL`** if not on localhost, upload a PDF with **`POST /documents/upload`**, then call **`POST /research`** or **`/research/stream`** with JSON **`{"query": "...", "document_ids": ["<document_id>"]}`**. Retrieved chunks are merged **before** web hits so they appear first in the source list. See **`.env.example`** for **`RAG_*`** tuning.

---

## How it works

Users can run **web-only** research, or **add uploaded PDFs**: `POST /documents/upload` stores chunked text in **Qdrant** with embeddings; sending **`document_ids`** on **`/research`** or **`/research/stream`** turns on **RAG** (semantic retrieval from those PDFs merged with web search). Qdrant must be running for upload/RAG; see **API endpoints** above.

1. The user submits a query (CLI or API), optionally after attaching one or more PDFs in the UI (or via the upload API).
2. The **Supervisor** uses deterministic rules to choose the next step.
3. The **Search Agent** builds 3 queries and fetches up to 9 primary DuckDuckGo hits per run (plus alternate-query / Wikipedia fills when a sub-query is sparse). If **`document_ids`** are set, **Qdrant** retrieval for uploaded PDFs is merged **before** web results.
4. **Research memory** (if any prior runs exist) is loaded into **`memory_context`** so analysis and writing can reference earlier topics without treating them as live web sources.
5. The **Analysis Agent** turns raw hits into structured analysis.
6. The **Writer Agent** drafts a markdown report (title, summary, findings, analysis, conclusion).
7. The **Critic Agent** returns APPROVED or REVISE (and increments `revision_count` on each critic pass).
8. On **REVISE**, the **Writer** revises using the critique, then **clears `critique`** so the supervisor routes to the **critic** again (not writer in a loop). After **two critic reviews** (`revision_count >= 2`), the supervisor ends the graph even if the report is still not approved (`supervisor.py`).
9. After a completed run, the pipeline **appends** the query and summary snippet to the JSON memory file (configurable via **`.env.example`**).
10. The **Evaluator** scores the report and returns JSON scores plus feedback.
11. When using the API, the final report is written to `/tmp/reports` and the JSON response includes scores and paths.

---

## License

MIT
