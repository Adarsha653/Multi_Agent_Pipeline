from langgraph.graph import StateGraph, END
from graph.state import AgentState
from agents.supervisor import supervisor_node
from agents.search_agent import search_agent_node
from agents.analysis_agent import analysis_agent_node
from agents.writer_agent import writer_agent_node
from agents.critic_agent import critic_agent_node
from utils.logger import PipelineLogger
from eval.evaluator import evaluate_report
import time

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

def run_pipeline(query: str):
    logger = PipelineLogger(query)
    start_time = time.time()

    initial_state = {
        'query': query,
        'messages': [],
        'search_results': [],
        'analysis': '',
        'report': '',
        'critique': '',
        'is_approved': False,
        'next_agent': '',
        'revision_count': 0
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

    scores = evaluate_report(query, result['report'], result['search_results'])

    print(f'\nTotal pipeline time: {total_time}s')
    print(f'Revisions made: {result["revision_count"]}')
    print(f'Approved: {result["is_approved"]}')
    logger.summary()

    return result, scores

if __name__ == '__main__':
    run_pipeline('What are the latest AI breakthroughs in 2025?')
