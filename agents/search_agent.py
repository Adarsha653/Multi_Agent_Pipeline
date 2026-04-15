from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from tools.search_tools import web_search
from graph.state import AgentState
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

def search_agent_node(state: AgentState) -> AgentState:
    print('Search Agent: generating search queries...')

    response = llm.invoke([
        SystemMessage(content='You are a research assistant. Given a topic, generate 3 simple plain English search queries (no boolean operators, no quotes, no AND/OR). Just natural phrases like a human would type into Google. Return as a numbered list, nothing else.'),
        HumanMessage(content=state['query'])
    ])

    lines = response.content.strip().split('\n')
    queries = [l.split('. ', 1)[-1].strip() for l in lines if l.strip()]

    all_results = []
    for q in queries[:3]:
        print(f'   Searching: {q}')
        results = web_search(q, max_results=3)
        all_results.extend(results)

    print(f'   Retrieved {len(all_results)} results')
    return {
        **state,
        'search_results': all_results,
        'messages': state['messages'] + [
            AIMessage(content=f"Search Agent retrieved {len(all_results)} results for: {state['query']}")
        ]
    }
