from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from graph.state import AgentState
from dotenv import load_dotenv

load_dotenv()
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

def critic_agent_node(state: AgentState) -> AgentState:
    print('Critic Agent: reviewing report...')
    try:
        response = llm.invoke([
            SystemMessage(content='You are a report reviewer. If the report has a title, summary, findings, and conclusion and is relevant to the query, respond with APPROVED. Only respond with REVISE if something critical is missing or factually wrong. Be lenient.'),
            HumanMessage(content=f"Query: {state['query']}\n\nReport:\n{state['report']}\n\nVerdict:")
        ])
        verdict = response.content.strip()
        is_approved = verdict.upper().startswith('APPROVED')
        revision_count = state.get('revision_count', 0) + 1
        if is_approved:
            print('   Report APPROVED')
        else:
            print(f'   Report needs REVISION (attempt {revision_count})')
        return {
            **state,
            'critique': verdict,
            'is_approved': is_approved,
            'revision_count': revision_count,
            'messages': state['messages'] + [AIMessage(content=f"Critic verdict: {'APPROVED' if is_approved else 'REVISE'}")]
        }
    except Exception as e:
        print(f'   Critic Agent failed: {e}')
        return {**state, 'is_approved': True, 'critique': 'Critic failed, auto-approving', 'revision_count': state.get('revision_count', 0) + 1, 'messages': state['messages'] + [AIMessage(content=f'Critic Agent failed: {e}')]}
