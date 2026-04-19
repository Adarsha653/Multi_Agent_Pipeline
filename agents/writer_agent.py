from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from graph.state import AgentState
from dotenv import load_dotenv

load_dotenv()
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

def writer_agent_node(state: AgentState) -> AgentState:
    critique = state.get('critique', '')
    is_revision = bool(critique)
    if is_revision:
        print('Writer Agent: revising report based on critique...')
    else:
        print('Writer Agent: generating report...')
    try:
        critique_section = f'\n\nPrevious critique to address:\n{critique}' if is_revision else ''
        response = llm.invoke([
            SystemMessage(content='You are a professional report writer. Write a clear structured report with sections: # Title, ## Executive Summary, ## Key Findings, ## Detailed Analysis, ## Conclusion.'),
            HumanMessage(content=f"Query: {state['query']}\n\nAnalysis:\n{state['analysis']}{critique_section}\n\nWrite the full report now.")
        ])
        print('   Report written')
        return {
            **state,
            'report': response.content,
            'messages': state['messages'] + [AIMessage(content=f"Writer completed {'revision' if is_revision else 'report'} for: {state['query']}")]
        }
    except Exception as e:
        print(f'   Writer Agent failed: {e}')
        return {**state, 'report': f'Report generation failed: {e}', 'messages': state['messages'] + [AIMessage(content=f'Writer Agent failed: {e}')]}
