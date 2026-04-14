from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from graph.state import AgentState
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

def critic_agent_node(state: AgentState) -> AgentState:
    print("Critic Agent: reviewing report...")

    response = llm.invoke([
        SystemMessage(content="You are a strict report reviewer. Evaluate the report on accuracy, completeness, clarity, and structure. If good enough respond with APPROVED followed by brief reason. If not respond with REVISE followed by specific actionable feedback."),
        HumanMessage(content=f"Query: {state['query']}\n\nAnalysis:\n{state['analysis']}\n\nReport:\n{state['report']}\n\nYour verdict:")
    ])

    verdict = response.content.strip()
    is_approved = verdict.upper().startswith("APPROVED")
    revision_count = state.get("revision_count", 0) + 1

    if is_approved:
        print("   Report APPROVED")
    else:
        print(f"   Report needs REVISION (attempt {revision_count})")

    return {
        **state,
        "critique": verdict,
        "is_approved": is_approved,
        "revision_count": revision_count,
        "messages": [
            AIMessage(content=f"Critic verdict: {'APPROVED' if is_approved else 'REVISE'}")
        ]
    }
