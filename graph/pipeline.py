import os
from typing import Any, Iterator

os.environ['LANGCHAIN_TRACING_V2'] = 'false'
from langgraph.graph import StateGraph, END
from graph.state import AgentState
from agents.supervisor import supervisor_node
from agents.search_agent import search_agent_node
from agents.analysis_agent import analysis_agent_node
from agents.writer_agent import writer_agent_node
from agents.critic_agent import critic_agent_node
from utils.logger import PipelineLogger
from eval.evaluator import evaluate_report
from utils.report_outcome import is_pipeline_failure_report, skipped_eval_scores
from utils.research_memory import format_memory_for_prompt, record_research_memory
import time

# User-facing SSE labels when supervisor routes to each worker (stream_mode="values" on next_agent change).
_STEP_LABELS: dict[str, str] = {
    'search_agent': 'Searching the web…',
    'analysis_agent': 'Analysing findings…',
    'writer_agent': 'Writing report…',
    'critic_agent': 'Reviewing report…',
}

def route(state: AgentState) -> str:
    return state.get('next_agent', 'END')

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node('supervisor', supervisor_node)
    graph.add_node('search_agent', search_agent_node)
    graph.add_node('analysis_agent', analysis_agent_node)
    graph.add_node('writer_agent', writer_agent_node)
    graph.add_node('critic_agent', critic_agent_node)
    graph.set_entry_point('supervisor')
    graph.add_conditional_edges('supervisor', route, {
        'search_agent': 'search_agent',
        'analysis_agent': 'analysis_agent',
        'writer_agent': 'writer_agent',
        'critic_agent': 'critic_agent',
        'END': END
    })
    graph.add_edge('search_agent', 'supervisor')
    graph.add_edge('analysis_agent', 'supervisor')
    graph.add_edge('writer_agent', 'supervisor')
    graph.add_edge('critic_agent', 'supervisor')
    return graph.compile()

def run_pipeline(query: str, document_ids: list[str] | None = None):
    logger = PipelineLogger(query)
    start_time = time.time()

    memory_context = format_memory_for_prompt(query)
    docs = [d.strip() for d in (document_ids or []) if d and str(d).strip()]
    initial_state = {
        'query': query,
        'document_ids': docs,
        'search_ran': False,
        'messages': [],
        'search_results': [],
        'analysis': '',
        'report': '',
        'critique': '',
        'is_approved': False,
        'next_agent': '',
        'revision_count': 0,
        'memory_context': memory_context,
    }

    pipeline = build_graph()
    print(f'\nRunning pipeline for: {query}\n')

    logger.start_agent('full_pipeline')
    result = pipeline.invoke(initial_state)
    total_time = round(time.time() - start_time, 2)
    logger.end_agent('full_pipeline', {'total_seconds': total_time})

    print('\n' + '='*60)
    print('FINAL REPORT')
    print('='*60)
    print(result['report'])

    if is_pipeline_failure_report(result.get('report')):
        scores = skipped_eval_scores()
    else:
        scores = evaluate_report(query, result['report'], result['search_results'])

    print(f'\nTotal pipeline time: {total_time}s')
    print(f'Revisions made: {result["revision_count"]}')
    print(f'Approved: {result["is_approved"]}')
    logger.summary()

    record_research_memory(query, result.get('report', ''))

    return result, scores


def iter_research_events(query: str, document_ids: list[str] | None = None) -> Iterator[dict[str, Any]]:
    """
    Run the graph once while yielding progress dicts for SSE.
    Yields: {"type": "step", "step": str, "message": str}, then {"type": "complete", ...}.
    """
    logger = PipelineLogger(query)
    start_time = time.time()
    memory_context = format_memory_for_prompt(query)
    docs = [d.strip() for d in (document_ids or []) if d and str(d).strip()]
    initial_state: AgentState = {
        'query': query,
        'document_ids': docs,
        'search_ran': False,
        'messages': [],
        'search_results': [],
        'analysis': '',
        'report': '',
        'critique': '',
        'is_approved': False,
        'next_agent': '',
        'revision_count': 0,
        'memory_context': memory_context,
    }
    pipeline = build_graph()
    print(f'\nRunning pipeline for: {query}\n')
    logger.start_agent('full_pipeline')

    prev_next = None
    final_state: dict[str, Any] = dict(initial_state)

    for state in pipeline.stream(initial_state, stream_mode='values'):
        final_state = state
        na = (state.get('next_agent') or '').strip()
        if na != prev_next and na in _STEP_LABELS:
            yield {'type': 'step', 'step': na, 'message': _STEP_LABELS[na]}
            prev_next = na
        elif na != prev_next:
            prev_next = na

    total_time = round(time.time() - start_time, 2)
    logger.end_agent('full_pipeline', {'total_seconds': total_time})

    yield {'type': 'step', 'step': 'evaluator', 'message': 'Scoring report…'}
    if is_pipeline_failure_report(final_state.get('report')):
        scores = skipped_eval_scores()
    else:
        scores = evaluate_report(query, final_state.get('report', ''), final_state.get('search_results') or [])

    print(f'\nTotal pipeline time: {total_time}s')
    print(f'Revisions made: {final_state.get("revision_count", 0)}')
    print(f'Approved: {final_state.get("is_approved", False)}')
    logger.summary()

    record_research_memory(query, final_state.get('report', ''))

    yield {
        'type': 'complete',
        'duration_seconds': total_time,
        'scores': scores,
        'result': {
            'report': final_state.get('report', ''),
            'is_approved': final_state.get('is_approved', False),
            'revision_count': final_state.get('revision_count', 0),
        },
    }


if __name__ == '__main__':
    run_pipeline('What are the latest AI breakthroughs in 2025?')
