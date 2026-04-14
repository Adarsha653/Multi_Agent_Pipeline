from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from graph.state import AgentState
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

def supervisor_node(state: AgentState) -> AgentState:
    """
    Decides which agent to call next based on current state.
    """
    system_prompt = """You are a supervisor orchestrating a research pipeline.
    
Given the current state of the task, decide which agent should act next.
Your options are: 'search_agent', 'analysis_agent', 'writer_agent', 'critic_agent', 'END'

Rules:
- If no search has been done yet → 'search_agent'
- If search is done but no analysis → 'analysis_agent'  
- If analysis is done but no report → 'writer_agent'
- If report exists but not approved → 'critic_agent'
- If approved OR revision_count >= 2 → 'END'

Respond with ONLY the agent name, nothing else."""

    user_message = f"""
Query: {state['query']}
Search done: {bool(state.get('search_results'))}
Analysis done: {bool(state.get('analysis'))}
Report done: {bool(state.get('report'))}
Approved: {state.get('is_approved', False)}
Revision count: {state.get('revision_count', 0)}
"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ])

    valid_agents = {"search_agent", "analysis_agent", "writer_agent", "critic_agent", "END"}
    next_agent = response.content.strip().split()[0].strip("`'\".,:;")
    if next_agent not in valid_agents:
        next_agent = "END"
    print(f"🎯 Supervisor → routing to: {next_agent}")

    return {**state, "next_agent": next_agent}