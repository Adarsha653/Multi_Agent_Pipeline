# Multi-Agent Research Pipeline

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-1.1.6-purple)
![Groq](https://img.shields.io/badge/Groq-Llama_3.3_70B-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.135.3-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

An autonomous multi-agent system that researches any topic end-to-end — searching the web, synthesizing findings, writing a structured report, reviewing it, and scoring it automatically. Built entirely on a free stack.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                     LangGraph Pipeline                  │
│                                                         │
│   ┌───────────┐                                        │
│   │Supervisor │ ◄──────────────────────────────┐       │
│   └─────┬─────┘                                │       │
│         │                                      │       │
│    ┌────▼────┐   ┌──────────┐   ┌──────────┐  │       │
│    │ Search  │──►│ Analysis │──►│  Writer  │──┤       │
│    │  Agent  │   │  Agent   │   │  Agent   │  │       │
│    └─────────┘   └──────────┘   └────┬─────┘  │       │
│                                       │        │       │
│                                  ┌────▼─────┐  │       │
│                                  │  Critic  │──┘       │
│                                  │  Agent   │  REVISE  │
│                                  └────┬─────┘          │
│                                       │ APPROVED       │
└───────────────────────────────────────┼────────────────┘
                                        │
                                        ▼
                              ┌──────────────────┐
                              │   Evaluator      │
                              │ (scores report)  │
                              └──────────────────┘
                                        │
                                        ▼
                              Final Report + Scores
```

---

## Agents

| Agent | Role |
|---|---|
| **Supervisor** | Reads current state and decides which agent runs next. Acts as the orchestrator — never does research itself. |
| **Search Agent** | Generates 3 targeted sub-queries and retrieves web results via DuckDuckGo. |
| **Analysis Agent** | Synthesizes raw search results into structured insights — key findings, themes, conflicts. |
| **Writer Agent** | Produces a formatted markdown report. On revision runs, incorporates critic feedback. |
| **Critic Agent** | Reviews the report and returns APPROVED or REVISE with specific feedback. |
| **Evaluator** | Scores the final report on relevance, completeness, clarity, and structure (1-10). |

---

## Tech Stack

| Layer | Tool | Cost |
|---|---|---|
| Agent Orchestration | LangGraph | Free |
| LLM Framework | LangChain | Free |
| LLM Model | Llama 3.3 70B via Groq | Free |
| Web Search | DuckDuckGo (ddgs) | Free |
| API | FastAPI | Free |
| Observability | LangSmith | Free tier |

**Total cost to run: $0**

---

## Project Structure

```
Multi_Agent_Pipeline/
├── agents/
│   ├── supervisor.py        # Orchestrator — routes between agents
│   ├── search_agent.py      # Web search and query generation
│   ├── analysis_agent.py    # Synthesizes search results
│   ├── writer_agent.py      # Report generation with revision support
│   └── critic_agent.py      # Report review and approval
├── graph/
│   ├── state.py             # Shared AgentState schema
│   └── pipeline.py          # LangGraph graph definition + run_pipeline()
├── tools/
│   └── search_tools.py      # DuckDuckGo web search wrapper
├── api/
│   └── main.py              # FastAPI backend
├── eval/
│   └── evaluator.py         # LLM-as-judge evaluation harness
├── utils/
│   └── logger.py            # Structured session logger
├── logs/                    # Session logs (auto-generated)
├── reports/                 # Saved markdown reports (auto-generated)
├── ui.html                  # Web interface
├── .env                     # API keys (never committed)
├── .gitignore
└── requirements.txt
```

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/Adarsha653/Multi_Agent_Pipeline.git
cd Multi_Agent_Pipeline
```

### 2. Create and activate virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Get free API keys
| Key | Where to get it |
|---|---|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) — free, no credit card |
| `LANGCHAIN_API_KEY` | [smith.langchain.com](https://smith.langchain.com) — free tier |

### 5. Create `.env` file
```env
GROQ_API_KEY=your_groq_key_here
LANGCHAIN_API_KEY=your_langsmith_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=multi-agent-pipeline
```

---

## Usage

### Run via terminal
```bash
python3 -m graph.pipeline
```

### Run via API
```bash
# Start the server
uvicorn api.main:app --reload --port 8000

# Run a research query
curl -X POST http://localhost:8000/research \
  -H 'Content-Type: application/json' \
  -d '{"query": "Latest AI breakthroughs in 2025"}'
```

### Run via Web UI
1. Start the API server (above)
2. Open `ui.html` in your browser
3. Type any research query and click **Research**

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/research` | Run the full pipeline for a query |
| `GET` | `/reports` | List all saved reports |
| `GET` | `/reports/{filename}` | Fetch a specific report |

### Example request
```json
POST /research
{
  "query": "What are the latest AI breakthroughs in 2025?"
}
```

### Example response
```json
{
  "query": "What are the latest AI breakthroughs in 2025?",
  "report": "# Latest AI Breakthroughs...\n## Executive Summary...",
  "scores": {
    "relevance": 9,
    "completeness": 8,
    "clarity": 8,
    "structure": 9,
    "overall": 8,
    "feedback": "Well structured report..."
  },
  "approved": true,
  "revisions": 1,
  "duration_seconds": 12.52,
  "report_file": "reports/report_20260420_092405.md"
}
```

---

## How It Works

1. **User submits a query** via terminal, API, or web UI
2. **Supervisor** reads the current state and routes to the Search Agent
3. **Search Agent** generates 3 plain English sub-queries and retrieves 9 web results via DuckDuckGo
4. **Analysis Agent** synthesizes results into key findings, themes, and conflicts
5. **Writer Agent** produces a structured markdown report with title, summary, findings, analysis, and conclusion
6. **Critic Agent** reviews the report — returns APPROVED or REVISE with feedback
7. If REVISE, Writer Agent incorporates the feedback and rewrites. Max 2 revision cycles.
8. **Evaluator** scores the final report on 4 dimensions (1-10) using LLM-as-judge
9. Report is saved to disk as a markdown file and returned via API

---

## Key Design Decisions

**Shared state architecture** — all agents read from and write to a single `AgentState` TypedDict. No agent talks to another directly — they communicate through state.

**Temperature 0 for all agents** — deterministic outputs are critical for reliable routing and structured responses (APPROVED/REVISE, JSON scores).

**Revision limit** — the pipeline hard stops after 2 revision cycles to prevent infinite loops, auto-approving if the critic keeps rejecting.

**Graceful error handling** — every agent catches exceptions and returns a safe fallback state rather than crashing the pipeline.

**LLM-as-judge evaluation** — the same LLM scores the final report on 4 dimensions, giving measurable quality signals across runs.

---

## License

MIT
