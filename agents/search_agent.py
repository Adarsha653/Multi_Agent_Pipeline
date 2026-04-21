from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from tools.search_tools import web_search, dedupe_and_trim_search_results, merge_search_results
from graph.state import AgentState
from dotenv import load_dotenv
from utils.groq_llm import chat_groq, user_message_for_groq_limit
from utils.rag_store import search_results_from_rag

load_dotenv()
llm = chat_groq()

def search_agent_node(state: AgentState) -> AgentState:
    print('Search Agent: generating search queries...')
    try:
        doc_ids = state.get('document_ids') or []
        rag_hits: list[dict] = []
        if doc_ids:
            print(f'   RAG: retrieving chunks for {len(doc_ids)} document(s)…')
            rag_hits = search_results_from_rag(state['query'], doc_ids)
            print(f'   RAG: {len(rag_hits)} chunk(s)')

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
        web_trimmed = dedupe_and_trim_search_results(all_results)
        combined = dedupe_and_trim_search_results(merge_search_results(rag_hits, web_trimmed))
        print(
            f'   Web: {raw_n} raw hits → {len(web_trimmed)} unique; '
            f'with RAG: {len(combined)} total (deduped, snippets trimmed)'
        )
        if not combined:
            print('   WARNING: No results retrieved — pipeline will continue with empty results')
        return {
            **state,
            'search_results': combined,
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
            'search_results': [],
            'search_ran': True,
            'messages': state['messages'] + [AIMessage(content=f'Search Agent failed: {user_msg}')],
        }
