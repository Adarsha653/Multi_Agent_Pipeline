# Multi-Agent Research Pipeline

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-1.1.6-purple)
![LangChain](https://img.shields.io/badge/LangChain-1.2.15-black)
![Groq](https://img.shields.io/badge/Groq-Llama_3.3_70B-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.135.3-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

An autonomous multi-agent system that researches any topic end-to-end — searching the web, synthesizing findings, writing a structured report, reviewing it, and scoring it automatically. Built on a free-tier friendly stack (Groq, DuckDuckGo, open-source libraries).

---

## Live Demo

Hugging Face Space: https://huggingface.co/spaces/AdarshaAryal653/multi-agent-pipeline

---

## Agents

| Agent | Role |
| --- | --- |
| Supervisor | Pure logic routing — decides which agent runs next (no LLM call) |
| Search Agent | Generates 3 diverse sub-queries (overview / recent / deeper angle) and up to 9 primary DDGS hits (3×3, deduped + trimmed); sparse sub-queries pull **alternate DDGS** phrasing and **Wikipedia** |
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
│   └── api_auth.py
├── tests/
│   ├── test_supervisor.py
│   ├── test_search_tools.py
│   ├── test_api_auth.py
│   ├── test_report_routes_auth.py
│   └── test_health_ready.py
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
| GET | `/reports` | Lists saved report filenames |
| GET | `/reports/{filename}` | Returns report metadata and markdown content |
| GET | `/reports/{filename}/download` | Downloads the report as a file |
| GET | `/health` | Liveness — process up (no auth, not rate-limited) |
| GET | `/ready` | Readiness — **503** if `GROQ_API_KEY` missing (no Groq call) |

Optional API protection (see `.env.example`): **`PIPELINE_API_KEY`** for **`POST /research`** and **`POST /research/stream`**; **`PIPELINE_REPORTS_API_KEY`** or the same pipeline key for **`GET /reports*`**. Rate limits use **`slowapi`** (configurable via env). The bundled **`ui.html`** does not send API keys—leave keys unset for a public browser demo, or use **`curl`** / a custom client with **`X-API-Key`** when keys are enabled.

---

## Improvements changelog (fixes 1–13)

| # | Area | What changed |
| --- | --- | --- |
| **1** | Revise loop | Writer clears **`critique`** after a successful draft so the supervisor returns to the **critic** instead of looping on the writer; revision cap remains two critic passes (`supervisor.py`). |
| **2** | Search quality | **Dedupe** by URL / fingerprint and **trim** long snippets in `tools/search_tools.py` before analysis. If DDGS returns **fewer than `WEB_SEARCH_SPARSE_THRESHOLD`** (default 3) hits for a sub-query, **retry DDGS** with a broader phrase, then **Wikipedia** (`action=opensearch`) to fill gaps. |
| **3** | Citations | **`[n]`** in-text markers plus **`## References`** in **APA 7th** (web) when web results exist; critic can **REVISE** if missing. |
| **4** | Structured LLM output | **Critic** and **evaluator** use Pydantic + **`with_structured_output`**, with string/JSON **fallbacks** if parsing fails. |
| **5** | Code hygiene | Removed unused **Groq** imports from **`supervisor.py`** (routing stays pure logic). |
| **6** | Resilience | Shared **`utils/groq_llm.py`** (**timeout**, **retries**); **DDGS** retries with backoff; **`POST /research`** returns generic **500** / **503** on timeout and logs server-side. |
| **7** | Tests | **`pytest`** + **`tests/test_supervisor.py`**, **`test_search_tools.py`**, **`test_api_auth.py`**. |
| **8** | API gate + limits | Optional **`PIPELINE_API_KEY`** on **`/research`**; **`slowapi`** per-IP limits. Send **`X-API-Key`** from **`curl`**, proxies, or custom clients when keys are set (the bundled **`ui.html`** does not collect keys). |
| **9** | CI | **`.github/workflows/ci.yml`** runs **`pytest`** on push/PR to **`main`** / **`master`**. |
| **10** | Report auth | Same key dependency on **`GET /reports`** routes when **`PIPELINE_API_KEY`** is set. |
| **11** | Ops | **`GET /health`** and **`GET /ready`** (exempt from rate limits). |
| **12** | Least privilege | Optional **`PIPELINE_REPORTS_API_KEY`** on report routes **or** **`PIPELINE_API_KEY`**; **`POST /research`** still **only** **`PIPELINE_API_KEY`**. Use **`curl`** / API clients with headers when keys are enabled. |
| **13** | Query planning | Search agent system prompt asks for **three distinct angles** (overview, recent, deeper) to reduce redundant queries. |
| **14** | Streaming UX | **`POST /research/stream`** emits **SSE** step events from **`iter_research_events`** in **`graph/pipeline.py`**; **`ui.html`** consumes the stream and shows live progress. |

*Not implemented (optional later): Redis-backed rate limits, dedicated fact-check agent.*

---

## How it works

1. The user submits a query (CLI or API).
2. The **Supervisor** uses deterministic rules to choose the next step.
3. The **Search Agent** builds 3 queries and fetches up to 9 primary DuckDuckGo hits per run (plus alternate-query / Wikipedia fills when a sub-query is sparse).
4. The **Analysis Agent** turns raw hits into structured analysis.
5. The **Writer Agent** drafts a markdown report (title, summary, findings, analysis, conclusion).
6. The **Critic Agent** returns APPROVED or REVISE (and increments `revision_count` on each critic pass).
7. On **REVISE**, the **Writer** revises using the critique, then **clears `critique`** so the supervisor routes to the **critic** again (not writer in a loop). After **two critic reviews** (`revision_count >= 2`), the supervisor ends the graph even if the report is still not approved (`supervisor.py`).
8. The **Evaluator** scores the report and returns JSON scores plus feedback.
9. When using the API, the final report is written to `/tmp/reports` and the JSON response includes scores and paths.

---

## License

MIT
