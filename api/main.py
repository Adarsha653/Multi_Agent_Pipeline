from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from graph.pipeline import run_pipeline
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os
import re
import time
from datetime import datetime

load_dotenv()

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

llm = ChatGroq(model='llama-3.3-70b-versatile', temperature=0)

def generate_filename(query: str) -> str:
    try:
        response = llm.invoke([
            SystemMessage(content='Generate a short 3-5 word kebab-case filename (lowercase, hyphens only, no spaces, no special chars, no extension) that summarizes the query. Examples: ai-breakthroughs-2025, quantum-computing-overview, electric-vehicle-trends. Return ONLY the filename, nothing else.'),
            HumanMessage(content=query)
        ])
        raw = response.content.strip().lower()
        clean = re.sub(r'[^a-z0-9-]', '', raw)[:60]
        return clean if clean else 'research-report'
    except:
        fallback = re.sub(r'[^a-z0-9]+', '-', query.lower())[:40].strip('-')
        return fallback if fallback else 'research-report'

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
    filename: str

@app.get('/')
def root():
    return {'status': 'running', 'message': 'Multi-Agent Research Pipeline API'}

@app.post('/research', response_model=ReportResponse)
def run_research(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail='Query cannot be empty')
    try:
        start = time.time()
        result, scores = run_pipeline(request.query)
        duration = round(time.time() - start, 2)
        base_name = generate_filename(request.query)
        filename = f'{base_name}.md'
        filepath = f'reports/{filename}'
        counter = 1
        while os.path.exists(filepath):
            filename = f'{base_name}-{counter}.md'
            filepath = f'reports/{filename}'
            counter += 1
        with open(filepath, 'w') as f:
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
            report_file=filepath,
            filename=filename
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

@app.get('/reports/{filename}/download')
def download_report(filename: str):
    path = f'reports/{filename}'
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail='Report not found')
    return FileResponse(path, media_type='text/markdown', filename=filename)
