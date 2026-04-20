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
| Search Agent | Generates 3 sub-queries and retrieves up to 9 web results via DuckDuckGo (3 queries × 3 results) |
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
| Web search | DuckDuckGo (`ddgs`) | Free |
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
│   └── logger.py
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
| GET | `/reports` | Lists saved report filenames |
| GET | `/reports/{filename}` | Returns report metadata and markdown content |
| GET | `/reports/{filename}/download` | Downloads the report as a file |

---

## How it works

1. The user submits a query (CLI or API).
2. The **Supervisor** uses deterministic rules to choose the next step.
3. The **Search Agent** builds 3 queries and fetches up to 9 DuckDuckGo results.
4. The **Analysis Agent** turns raw hits into structured analysis.
5. The **Writer Agent** drafts a markdown report (title, summary, findings, analysis, conclusion).
6. The **Critic Agent** returns APPROVED or REVISE.
7. On REVISE, the writer can incorporate feedback; the supervisor stops the graph after enough critic rounds (revision budget enforced in `supervisor.py`).
8. The **Evaluator** scores the report and returns JSON scores plus feedback.
9. When using the API, the final report is written to `/tmp/reports` and the JSON response includes scores and paths.

---

## License

MIT
