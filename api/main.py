from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from graph.pipeline import run_pipeline
import os
import json
from datetime import datetime

app = FastAPI(
    title='Multi-Agent Research Pipeline',
    description='Autonomous research pipeline powered by LangGraph and Groq',
    version='1.0.0'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

os.makedirs('reports', exist_ok=True)

class QueryRequest(BaseModel):
    query: str

class ReportResponse(BaseModel):
    query: str
    report: str
    scores: dict
    approved: bool
    revisions: int
    duration_seconds: float
    report_file: str

@app.get('/')
def root():
    return {'status': 'running', 'message': 'Multi-Agent Research Pipeline API'}

@app.post('/research', response_model=ReportResponse)
def run_research(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail='Query cannot be empty')
    try:
        import time
        start = time.time()
        result, scores = run_pipeline(request.query)
        duration = round(time.time() - start, 2)
        filename = f"reports/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(filename, 'w') as f:
            f.write(f'# Query\n{request.query}\n\n')
            f.write(result['report'])
            f.write(f'\n\n---\n')
            f.write(f'**Approved:** {result["is_approved"]}\n')
            f.write(f'**Revisions:** {result["revision_count"]}\n')
            f.write(f'**Duration:** {duration}s\n')
        return ReportResponse(
            query=request.query,
            report=result['report'],
            scores=scores,
            approved=result['is_approved'],
            revisions=result['revision_count'],
            duration_seconds=duration,
            report_file=filename
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/reports')
def list_reports():
    files = os.listdir('reports')
    reports = [f for f in files if f.endswith('.md')]
    return {'reports': sorted(reports, reverse=True)}

@app.get('/reports/{filename}')
def get_report(filename: str):
    path = f'reports/{filename}'
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail='Report not found')
    with open(path, 'r') as f:
        content = f.read()
    return {'filename': filename, 'content': content}
