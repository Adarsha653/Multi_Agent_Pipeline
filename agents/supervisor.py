from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from graph.state import AgentState
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

def supervisor_node(state: AgentState) -> AgentState:
    revision_count = state.get('revision_count', 0)

    if revision_count >= 2:
        print('Supervisor -> routing to: END (revision limit reached)')
        return {**state, 'next_agent': 'END'}

    system_prompt = (
        'You are a supervisor orchestrating a research pipeline.\n\n'
        'Decide which agent should act next. Options: search_agent, analysis_agent, writer_agent, critic_agent, END\n\n'
        'Rules:\n'
        '- No search done yet -> search_agent\n'
        '- Search done, no analysis -> analysis_agent\n'
        '- Analysis done, no report -> writer_agent\n'
        '- Report exists, not approved, critique exists -> writer_agent\n'
        '- Report revised, not approved -> critic_agent\n'
        '- Approved -> END\n\n'
        'Respond with ONLY the agent name.'
    )

    user_message = (
        f"Query: {state['query']}\n"
        f"Search done: {bool(state.get('search_results'))}\n"
        f"Analysis done: {bool(state.get('analysis'))}\n"
        f"Report done: {bool(state.get('report'))}\n"
        f"Critique exists: {bool(state.get('critique'))}\n"
        f"Approved: {state.get('is_approved', False)}\n"
        f"Revision count: {revision_count}\n"
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ])

    next_agent = response.content.strip()
    print(f'Supervisor -> routing to: {next_agent}')
    return {**state, 'next_agent': next_agent}
