import time

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from graph.state import AgentState
from dotenv import load_dotenv
from utils.groq_llm import chat_groq, user_message_for_groq_limit
from utils.agent_timing import append_step_duration

load_dotenv()
llm = chat_groq()

def analysis_agent_node(state: AgentState) -> AgentState:
    print('Analysis Agent: synthesizing search results...')
    t0 = time.perf_counter()
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
        confidence_block = (
            '\n\nAfter the main synthesis, add a section exactly titled `## Source support overview` with three subheadings:\n'
            '- **Strong (multiple sources):** bullet themes or claims that clearly rest on **two or more distinct** source numbers (e.g. [1] and [3] from different URLs).\n'
            '- **Single-source:** bullets that rely on **only one** source index throughout.\n'
            '- **Limited or conflicting:** bullets where evidence is thin, sources disagree, or fewer than two sources exist in total.\n'
            'If there are no search results, write "N/A — no retrieved sources" under each subheading.'
            if has_sources else
            '\n\nAdd `## Source support overview` stating that no retrieved sources were available (N/A under each subheading).'
        )
        response = llm.invoke([
            SystemMessage(content=(
                'You are an expert research analyst. Synthesize the provided material into: key findings, common themes, notable details, and conflicting information where relevant. '
                + ('Every factual bullet or paragraph that rests on the search results MUST end with one or more bracket citations like [1] or [1][3] using ONLY the source indices given below (no other numbers). Do not cite sources you were not given.'
                   if has_sources else
                   'There are no retrieved sources; write clearly that evidence is limited and do not use [n] source tags.')
                + confidence_block
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
            **append_step_duration(state, 'analysis_agent', t0),
            'analysis': response.content,
            'messages': state['messages'] + [AIMessage(content=f"Analysis complete for: {state['query']}")]
        }
    except Exception as e:
        print(f'   Analysis Agent failed: {e}')
        user_msg = user_message_for_groq_limit(e)
        return {
            **state,
            **append_step_duration(state, 'analysis_agent', t0),
            'analysis': f'Analysis failed: {user_msg}',
            'messages': state['messages'] + [AIMessage(content=f'Analysis Agent failed: {user_msg}')],
        }
