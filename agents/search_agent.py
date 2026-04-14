from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from tools.search_tools import web_search
from graph.state import AgentState

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

def search_agent_node(state: AgentState) -> AgentState:
    """
    Breaks down the query into sub-queries and retrieves web results.
    """
    print("🔍 Search Agent: generating search queries...")

    # Ask LLM to generate targeted sub-queries
    response = llm.invoke([
        SystemMessage(content="You are a research assistant. Given a query, generate 3 targeted search queries to gather comprehensive information. Return them as a numbered list, nothing else."),
        HumanMessage(content=state["query"])
    ])

    # Parse sub-queries
    lines = response.content.strip().split("\n")
    queries = [l.split(". ", 1)[-1].strip() for l in lines if l.strip()]

    # Run searches
    all_results = []
    for q in queries[:3]:
        print(f"   🔎 Searching: {q}")
        results = web_search(q, max_results=3)
        all_results.extend(results)

    print(f"   ✅ Retrieved {len(all_results)} results")

    return {
        **state,
        "search_results": all_results,
        "messages": [
            AIMessage(content=f"Search Agent retrieved {len(all_results)} results for query: {state['query']}")
        ]
    }