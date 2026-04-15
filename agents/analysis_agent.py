from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from graph.state import AgentState
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

def analysis_agent_node(state: AgentState) -> AgentState:
    print("Analysis Agent: synthesizing search results...")

    search_context = ""
    for i, r in enumerate(state["search_results"]):
        search_context += f"Source {i+1}: {r['title']}\n{r['content']}\nURL: {r['url']}\n\n"

    response = llm.invoke([
        SystemMessage(content="You are an expert research analyst. Given search results, synthesize the most important insights. Include: 1) Key findings 2) Common themes 3) Notable details 4) Conflicting information. Cite source numbers. Respond in English only, even if some sources are not in English."),
        HumanMessage(content=f"Query: {state['query']}\n\nSearch Results:\n{search_context}\n\nProvide structured analysis.")
    ])

    print("   Analysis complete")
    return {
        **state,
        "analysis": response.content,
        "messages": [
            AIMessage(content=f"Analysis Agent completed synthesis for: {state['query']}")
        ]
    }
