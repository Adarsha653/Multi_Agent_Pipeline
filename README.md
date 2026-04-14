# Multi Agent Pipeline

A LangGraph-based research workflow that routes tasks across multiple specialized agents:

- `supervisor` decides what should run next
- `search_agent` gathers web results
- `analysis_agent` synthesizes findings
- `writer_agent` drafts a report
- `critic_agent` reviews and approves/rejects output

## Project Structure

- `graph/` - state schema and pipeline graph
- `agents/` - agent node implementations
- `tools/` - helper tools (search, retrieval, etc.)
- `api/` - API entrypoint (if you expose the pipeline)
- `eval/` - evaluation scripts

## Requirements

- Python 3.10+
- A Groq API key

## Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here

# Optional: LangSmith tracing
# LANGSMITH_TRACING=true
# LANGSMITH_API_KEY=your_langsmith_api_key_here
# LANGSMITH_PROJECT=multi-agent-pipeline
```

## Run the Pipeline

From the project root:

```bash
python3 graph/pipeline.py
```

You should see routing logs like:

- `Supervisor -> search_agent`
- `Supervisor -> analysis_agent`
- `Supervisor -> writer_agent`
- `Supervisor -> critic_agent`
- `Supervisor -> END`

## Current Flow

The graph currently executes:

1. `supervisor`
2. `search_agent`
3. `analysis_agent`
4. `writer_agent`
5. `critic_agent`
6. loop back to `supervisor` until approved or max revisions reached

## Notes

- The supervisor output is validated before routing; unknown values safely end the graph.
- `messages` state uses LangGraph reducer semantics (`operator.add`), and each node appends only new messages.
- If search providers/rate limits change, update `tools/search_tools.py`.
