from graph.state import AgentState
from utils.report_outcome import is_pipeline_failure_report


def supervisor_node(state: AgentState) -> AgentState:
    revision_count = state.get('revision_count', 0)
    search_done = bool(state.get('search_ran'))
    analysis_done = bool(state.get('analysis'))
    report_done = bool(state.get('report'))
    # Writer clears critique after each successful draft so this becomes False again → critic_agent.
    critique_done = bool(state.get('critique'))
    is_approved = state.get('is_approved', False)

    # Do not send error stubs to the critic or mark them as successful products.
    if is_pipeline_failure_report(state.get('report')):
        print('Supervisor -> routing to: END (report or analysis step failed)')
        return {**state, 'next_agent': 'END', 'is_approved': False}

    # Hard rules — no LLM needed, pure logic
    if is_approved:
        next_agent = 'END'
    # revision_count increments once per critic visit; >= 2 means at most two critic reviews, then stop.
    elif revision_count >= 2:
        next_agent = 'END'
    elif not search_done:
        next_agent = 'search_agent'
    elif not analysis_done:
        next_agent = 'analysis_agent'
    elif not report_done:
        next_agent = 'writer_agent'
    elif not critique_done:
        next_agent = 'critic_agent'
    elif not is_approved and critique_done:
        next_agent = 'writer_agent'
    else:
        next_agent = 'END'

    print(f'Supervisor -> routing to: {next_agent}')
    return {**state, 'next_agent': next_agent}
