from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from graph.state import AgentState
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

def writer_agent_node(state: AgentState) -> AgentState:
    print("Writer Agent: generating report...")

    response = llm.invoke([
        SystemMessage(content="You are a professional report writer. Write a clear structured report with these sections: # Title, ## Executive Summary, ## Key Findings, ## Detailed Analysis, ## Conclusion. Use professional but accessible tone."),
        HumanMessage(content=f"Query: {state['query']}\n\nAnalysis:\n{state['analysis']}\n\nWrite the full report now.")
    ])

    print("   Report written")
    return {
        **state,
        "report": response.content,
        "messages": [
            AIMessage(content=f"Writer Agent completed report for: {state['query']}")
        ]
    }
