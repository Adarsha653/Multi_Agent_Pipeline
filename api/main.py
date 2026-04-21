import json
from datetime import datetime

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
import os as _os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import Literal

from pydantic import BaseModel, Field
from graph.pipeline import iter_research_events, run_pipeline
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import logging
import os
import re
import time

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from utils.api_auth import pipeline_api_key_dependency, reports_api_key_dependency
from utils.groq_llm import chat_groq, is_groq_rate_or_token_limit, user_message_for_groq_limit
from utils.report_outcome import is_pipeline_failure_report, saved_report_markdown_is_failure

load_dotenv()

logger = logging.getLogger(__name__)

RESEARCH_RATE_LIMIT = os.getenv('RESEARCH_RATE_LIMIT', '30/minute')
READ_REPORTS_RATE_LIMIT = os.getenv('READ_REPORTS_RATE_LIMIT', '120/minute')

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title='Multi-Agent Research Pipeline',
    description='Autonomous research pipeline powered by LangGraph and Groq',
    version='1.0.0',
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

os.makedirs('/tmp/reports', exist_ok=True)

llm = chat_groq()


def generate_filename(query: str) -> str:
    try:
        response = llm.invoke([
            SystemMessage(content='Generate a short 3-5 word kebab-case filename (lowercase, hyphens only, no spaces, no special chars, no extension) that summarizes the query. Examples: ai-breakthroughs-2025, quantum-computing-overview, electric-vehicle-trends. Return ONLY the filename, nothing else.'),
            HumanMessage(content=query)
        ])
        raw = response.content.strip().lower()
        clean = re.sub(r'[^a-z0-9-]', '', raw)[:60]
        return clean if clean else 'research-report'
    except Exception:
        logger.warning('generate_filename LLM failed; using slug fallback', exc_info=True)
        fallback = re.sub(r'[^a-z0-9]+', '-', query.lower())[:40].strip('-')
        return fallback if fallback else 'research-report'


def _persist_report(
    query: str,
    report: str,
    is_approved: bool,
    revision_count: int,
    duration: float,
) -> tuple[str, str]:
    """Write markdown to /tmp/reports; returns (filepath, filename)."""
    base_name = generate_filename(query)
    filename = f'{base_name}.md'
    filepath = f'/tmp/reports/{filename}'
    counter = 1
    while os.path.exists(filepath):
        filename = f'{base_name}-{counter}.md'
        filepath = f'/tmp/reports/{filename}'
        counter += 1
    with open(filepath, 'w') as f:
        f.write(f'# Query\n{query}\n\n')
        f.write(report)
        f.write('\n\n---\n')
        f.write(f'**Approved:** {is_approved}\n')
        f.write(f'**Revisions:** {revision_count}\n')
        f.write(f'**Duration:** {duration}s\n')
    return filepath, filename


ReportFormat = Literal['markdown', 'bullets', 'executive_only', 'full_detailed']


class QueryRequest(BaseModel):
    query: str
    report_format: ReportFormat = 'markdown'


class ReportResponse(BaseModel):
    query: str
    report: str
    scores: dict
    approved: bool
    revisions: int
    duration_seconds: float
    report_file: str
    filename: str
    agent_steps: list[dict] = Field(default_factory=list)
    report_format: ReportFormat = 'markdown'


@app.get('/', response_class=HTMLResponse)
def root():
    ui_path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'ui.html')
    with open(ui_path, 'r') as f:
        return f.read()


@app.get('/health')
@limiter.exempt
def health():
    """Fix #11 — process is up (for load balancers / Spaces / k8s liveness)."""
    return {'status': 'ok', 'service': 'multi-agent-research-pipeline'}


@app.get('/ready')
@limiter.exempt
def ready():
    """Fix #11 — required env present (readiness; does not call Groq)."""
    if not (os.getenv('GROQ_API_KEY') or '').strip():
        raise HTTPException(
            status_code=503,
            detail='GROQ_API_KEY is not configured',
        )
    return {'status': 'ready', 'groq': 'configured'}


@app.post('/research', response_model=ReportResponse)
@limiter.limit(RESEARCH_RATE_LIMIT)
def run_research(
    request: Request,
    payload: QueryRequest,
    _auth: None = Depends(pipeline_api_key_dependency),
):
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail='Query cannot be empty')
    try:
        start = time.time()
        result, scores = run_pipeline(payload.query, payload.report_format)
        duration = round(time.time() - start, 2)
        if is_pipeline_failure_report(result.get('report')):
            filepath, filename = '', ''
        else:
            filepath, filename = _persist_report(
                payload.query,
                result['report'],
                result['is_approved'],
                result['revision_count'],
                duration,
            )
        return ReportResponse(
            query=payload.query,
            report=result['report'],
            scores=scores,
            approved=result['is_approved'],
            revisions=result['revision_count'],
            duration_seconds=duration,
            report_file=filepath,
            filename=filename,
            agent_steps=result.get('agent_steps') or [],
            report_format=payload.report_format,
        )
    except Exception as e:
        logger.exception('POST /research failed')
        if is_groq_rate_or_token_limit(e):
            raise HTTPException(
                status_code=503,
                detail=user_message_for_groq_limit(e),
            ) from None
        detail = str(e).lower()
        if 'timeout' in detail or 'timed out' in detail:
            raise HTTPException(
                status_code=503,
                detail='The research request timed out. Try again or use a shorter query.',
            ) from None
        raise HTTPException(
            status_code=500,
            detail='Research pipeline failed. See server logs for details.',
        ) from None


@app.post('/research/stream')
@limiter.limit(RESEARCH_RATE_LIMIT)
def research_stream(
    request: Request,
    payload: QueryRequest,
    _auth: None = Depends(pipeline_api_key_dependency),
):
    """Server-Sent Events: live step labels, then one `complete` event (same fields as POST /research)."""
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail='Query cannot be empty')

    def event_generator():
        pending_complete = None
        try:
            for ev in iter_research_events(payload.query, payload.report_format):
                if ev.get('type') == 'complete':
                    pending_complete = ev
                    continue
                yield f'data: {json.dumps(ev)}\n\n'
            if pending_complete:
                r = pending_complete['result']
                if is_pipeline_failure_report(r.get('report')):
                    path, fname = '', ''
                else:
                    path, fname = _persist_report(
                        payload.query,
                        r['report'],
                        r['is_approved'],
                        r['revision_count'],
                        pending_complete['duration_seconds'],
                    )
                out = {
                    'type': 'complete',
                    'query': payload.query,
                    'report_format': payload.report_format,
                    'report': r['report'],
                    'scores': pending_complete['scores'],
                    'approved': r['is_approved'],
                    'revisions': r['revision_count'],
                    'duration_seconds': pending_complete['duration_seconds'],
                    'report_file': path,
                    'filename': fname,
                    'agent_steps': r.get('agent_steps') or [],
                }
                yield f'data: {json.dumps(out)}\n\n'
        except Exception as e:
            logger.exception('POST /research/stream failed')
            if is_groq_rate_or_token_limit(e):
                detail = user_message_for_groq_limit(e)
            else:
                low = str(e).lower()
                detail = (
                    'The research request timed out. Try again or use a shorter query.'
                    if 'timeout' in low or 'timed out' in low
                    else 'Research pipeline failed. See server logs for details.'
                )
            yield f'data: {json.dumps({"type": "error", "detail": detail})}\n\n'

    return StreamingResponse(
        event_generator(),
        media_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',
        },
    )


@app.get('/reports')
@limiter.limit(READ_REPORTS_RATE_LIMIT)
def list_reports(
    request: Request,
    _auth: None = Depends(reports_api_key_dependency),
):
    """Newest first by file mtime; each item includes local wall-clock `generated_at`."""
    base = '/tmp/reports'
    rows: list[tuple[float, dict[str, str]]] = []
    try:
        names = os.listdir(base)
    except OSError:
        return {'reports': []}
    for name in names:
        if not name.endswith('.md'):
            continue
        path = os.path.join(base, name)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        try:
            with open(path, encoding='utf-8', errors='replace') as fp:
                head = fp.read(12288)
        except OSError:
            continue
        if saved_report_markdown_is_failure(head):
            continue
        dt = datetime.fromtimestamp(mtime)
        rows.append(
            (
                mtime,
                {
                    'filename': name,
                    'generated_at': dt.strftime('%Y-%m-%d %H:%M:%S'),
                },
            )
        )
    rows.sort(key=lambda r: r[0], reverse=True)
    return {'reports': [r[1] for r in rows]}


@app.get('/reports/{filename}')
@limiter.limit(READ_REPORTS_RATE_LIMIT)
def get_report(
    request: Request,
    filename: str,
    _auth: None = Depends(reports_api_key_dependency),
):
    path = f'/tmp/reports/{filename}'
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail='Report not found')
    with open(path, 'r') as f:
        content = f.read()
    return {'filename': filename, 'content': content}


@app.get('/reports/{filename}/download')
@limiter.limit(READ_REPORTS_RATE_LIMIT)
def download_report(
    request: Request,
    filename: str,
    _auth: None = Depends(reports_api_key_dependency),
):
    path = f'/tmp/reports/{filename}'
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail='Report not found')
    return FileResponse(path, media_type='text/markdown', filename=filename)
