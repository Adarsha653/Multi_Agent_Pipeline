import time

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from tools.search_tools import web_search, dedupe_and_trim_search_results
from graph.state import AgentState
from dotenv import load_dotenv
from utils.groq_llm import chat_groq, user_message_for_groq_limit
from utils.agent_timing import append_step_duration

load_dotenv()
llm = chat_groq()

def search_agent_node(state: AgentState) -> AgentState:
    print('Search Agent: generating search queries...')
    t0 = time.perf_counter()
    try:
        response = llm.invoke([
            SystemMessage(content=(
                'You are a research assistant. Given a topic, generate exactly 3 simple plain-English web search queries as a numbered list. '
                'Use three different angles: (1) overview or definition, (2) recent developments or current year in the topic, '
                '(3) limitations, debate, risks, or a deeper technical angle. Avoid near-duplicate wording across the three. '
                'No boolean operators, no quotes, no AND/OR. Natural phrases only. Output only the numbered list.'
            )),
            HumanMessage(content=state['query'])
        ])
        lines = response.content.strip().split('\n')
        queries = [l.split('. ', 1)[-1].strip() for l in lines if l.strip()]
        all_results = []
        for q in queries[:3]:
            print(f'   Searching: {q}')
            try:
                results = web_search(q, max_results=3)
                all_results.extend(results)
            except Exception as e:
                print(f'   Search failed for query [{q}]: {e}')
        raw_n = len(all_results)
        all_results = dedupe_and_trim_search_results(all_results)
        print(f'   Retrieved {raw_n} raw hits → {len(all_results)} unique (deduped, snippets trimmed)')
        if not all_results:
            print('   WARNING: No results retrieved — pipeline will continue with empty results')
        return {
            **state,
            **append_step_duration(state, 'search_agent', t0),
            'search_results': all_results,
            'search_ran': True,
            'messages': state['messages'] + [
                AIMessage(content=f"Search Agent retrieved {len(combined)} results for: {state['query']}")
            ],
        }
    except Exception as e:
        print(f'   Search Agent failed: {e}')
        user_msg = user_message_for_groq_limit(e)
        return {
            **state,
            **append_step_duration(state, 'search_agent', t0),
            'search_results': [],
            'search_ran': True,
            'messages': state['messages'] + [AIMessage(content=f'Search Agent failed: {user_msg}')],
        }
