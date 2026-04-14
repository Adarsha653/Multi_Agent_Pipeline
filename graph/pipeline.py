import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from langgraph.graph import StateGraph, END
from graph.state import AgentState
from agents.supervisor import supervisor_node
from agents.search_agent import search_agent_node
from agents.analysis_agent import analysis_agent_node
from agents.writer_agent import writer_agent_node
from agents.critic_agent import critic_agent_node

def route(state: AgentState) -> str:
    """Router function that validates next_agent before routing."""
    valid_routes = {"search_agent", "analysis_agent", "writer_agent", "critic_agent", "END"}
    next_agent = state.get("next_agent", "END")
    return next_agent if next_agent in valid_routes else "END"

def build_graph():
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("search_agent", search_agent_node)
    graph.add_node("analysis_agent", analysis_agent_node)
    graph.add_node("writer_agent", writer_agent_node)
    graph.add_node("critic_agent", critic_agent_node)

    # Entry point
    graph.set_entry_point("supervisor")

    # Supervisor routes to agents
    graph.add_conditional_edges(
        "supervisor",
        route,
        {
            "search_agent": "search_agent",
            "analysis_agent": "analysis_agent",
            "writer_agent": "writer_agent",
            "critic_agent": "critic_agent",
            "END": END
        }
    )

    # After each agent, return to supervisor
    graph.add_edge("search_agent", "supervisor")
    graph.add_edge("analysis_agent", "supervisor")
    graph.add_edge("writer_agent", "supervisor")
    graph.add_edge("critic_agent", "supervisor")

    return graph.compile()

# Test run
if __name__ == "__main__":
    pipeline = build_graph()

    initial_state = {
        "query": "What are the latest AI breakthroughs in 2025?",
        "messages": [],
        "search_results": [],
        "analysis": "",
        "report": "",
        "critique": "",
        "is_approved": False,
        "next_agent": "",
        "revision_count": 0
    }

    result = pipeline.invoke(initial_state)
    print("\n📦 Search Results Preview:")
    for r in result["search_results"][:2]:
        print(f"  - {r['title']}")