from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from graph.state import AgentState
from dotenv import load_dotenv
from utils.groq_llm import chat_groq, user_message_for_groq_limit

load_dotenv()
llm = chat_groq()

def analysis_agent_node(state: AgentState) -> AgentState:
    print('Analysis Agent: synthesizing search results...')
    try:
        search_context = ''
        for i, r in enumerate(state['search_results']):
            search_context += f"Source {i+1}: {r['title']}\n{r['content']}\nURL: {r['url']}\n\n"
        has_sources = bool(state['search_results'])
        if not search_context:
            search_context = 'No search results available. Use general knowledge only; do not invent source numbers or URLs.'
        mem = (state.get('memory_context') or '').strip()
        memory_block = ''
        if mem:
            memory_block = (
                '\n\nEarlier research from past runs in this app (not live web sources — never use [n] for these):\n'
                f'{mem}\n\n'
                'When it genuinely helps, you may relate the current query to those prior themes with cautious wording '
                '(e.g. building on earlier work on EVs). Only the numbered Search Results below may receive [n] citations.\n'
            )
        response = llm.invoke([
            SystemMessage(content=(
                'You are an expert research analyst. Synthesize the provided material into: key findings, common themes, notable details, and conflicting information where relevant. '
                + ('Every factual bullet or paragraph that rests on the search results MUST end with one or more bracket citations like [1] or [1][3] using ONLY the source indices given below (no other numbers). Do not cite sources you were not given.'
                   if has_sources else
                   'There are no retrieved sources; write clearly that evidence is limited and do not use [n] source tags.')
            )),
            HumanMessage(
                content=(
                    f"Query: {state['query']}{memory_block}\n\nSearch Results:\n{search_context}\n\nProvide structured analysis."
                )
            ),
        ])
        print('   Analysis complete')
        return {
            **state,
            'analysis': response.content,
            'messages': state['messages'] + [AIMessage(content=f"Analysis complete for: {state['query']}")]
        }
    except Exception as e:
        print(f'   Analysis Agent failed: {e}')
        user_msg = user_message_for_groq_limit(e)
        return {
            **state,
            'analysis': f'Analysis failed: {user_msg}',
            'messages': state['messages'] + [AIMessage(content=f'Analysis Agent failed: {user_msg}')],
        }
