from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from graph.state import AgentState
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

def supervisor_node(state: AgentState) -> AgentState:
    revision_count = state.get('revision_count', 0)
    search_done = bool(state.get('search_results'))
    analysis_done = bool(state.get('analysis'))
    report_done = bool(state.get('report'))
    critique_done = bool(state.get('critique'))
    is_approved = state.get('is_approved', False)

    # Hard rules — no LLM needed, pure logic
    if is_approved:
        next_agent = 'END'
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
