from datetime import datetime

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from graph.state import AgentState
from dotenv import load_dotenv
from utils.groq_llm import chat_groq, user_message_for_groq_limit

load_dotenv()
llm = chat_groq()


def _apa_retrieved_phrase() -> str:
    """APA 'Retrieved Month Day, Year' segment for Wikipedia (access date = pipeline run)."""
    d = datetime.now()
    return f'{d.strftime("%B")} {d.day}, {d.year}'


def _format_source_catalog(results: list) -> tuple[str, str]:
    """Numbered list for prompts + short rules fragment."""
    if not results:
        return '', (
            'No web sources were retrieved. Write the report without [n] reference markers and without a ## References section; '
            'state clearly when claims are not backed by retrieved URLs.'
        )
    lines = []
    any_upload = False
    for i, r in enumerate(results, start=1):
        title = (r.get('title') or 'Untitled').strip()
        url = (r.get('url') or '').strip()
        if url.lower().startswith('upload://'):
            any_upload = True
        if url:
            wiki = 'wikipedia.org' in url.lower()
            hint = ' (use Wikipedia contributors … In Wikipedia … Retrieved … from URL)' if wiki else ''
            lines.append(f'[{i}] {title} — {url}{hint}')
        else:
            lines.append(f'[{i}] {title}')
    catalog = '\n'.join(lines)
    retrieved = _apa_retrieved_phrase()
    upload_block = ''
    if any_upload:
        upload_block = f'''

---

**Uploaded PDF** (catalog URL begins with `upload://` — internal user upload, not a public webpage):
Use gray-literature style. Prefer the filename or title fragment from the catalog line for the work title in sentence case.
Pattern: Author or Organization. (n.d.). Title from catalog in sentence case [Unpublished PDF]. Retrieved {retrieved}, from the exact upload:// URL shown in the catalog (verbatim).
If no author is known, use the PDF filename from the catalog as the author-like element before the date.
Do not invent an https:// URL for these entries.
'''
    rules = f'''Use ONLY in-text reference numbers [1] through [{len(results)}] from the catalog. Where evidence comes from those sources, end the sentence with the correct [n] markers (e.g. ... claim. [2]).
After ## Conclusion, add ## References as a markdown ordered list (`1.`, `2.`, …). List order must match in-text [1], [2], … (first source cited = item 1). Each item must follow **APA 7th edition** for the source type — never semicolon-separated catalog dumps.

**Access date for this report (use verbatim for Wikipedia "Retrieved" lines):** Retrieved {retrieved}, from

---

**Wikipedia** (any catalog URL whose host contains `wikipedia.org`):
Use exactly this pattern (article date from the page when inferable; else use `(n.d.)` for the article parenthetical):
`Wikipedia contributors. (Year, Month Day). Title of article in sentence case. In Wikipedia. Retrieved {retrieved}, from https://...`
Example shape: `Wikipedia contributors. (2024, March 12). Quantum computing. In Wikipedia. Retrieved {retrieved}, from https://en.wikipedia.org/wiki/Quantum_computing`
Do **not** use `Wikipedia.` alone as author. Always include **In Wikipedia.** before the Retrieved line. Do not add a period after the URL.

---

**General website / webpage** (URLs that are not Wikipedia):
`Author, A. A. (Year, Month Day). Title of page in sentence case. Site or publication name. https://...`
If there is no named person: `Organization Name. (Year, Month Day). Title of page in sentence case. https://...` (organization doubles as site when appropriate).
Use `(Year).` or `(n.d.).` when month/day are unknown. Title and site names use sentence case; end the title sentence with a period before the site name or URL as APA allows for standalone webpages.

---

**Other types** (only if the catalog URL clearly matches): news outlet → `Author. (Y, M D). Article title. Outlet Name. URL`; YouTube → channel as author, `[Video]`, `YouTube.`, URL. Otherwise default to the general webpage pattern.

---

**Wrong:** `scribd.com; (n.d.); title; url` or `Wikipedia. (2024, March 12). Quantum computing. Retrieved ...` (missing *Wikipedia contributors*, missing *In Wikipedia.*).
**Right (non-wiki):** `TechWorld. (2023, June 5). What is machine learning? TechWorld. https://example.com` (adapt author/site from the catalog).
Do not invent URLs; one list item per cited catalog index.{upload_block}'''
    return catalog, rules


def writer_agent_node(state: AgentState) -> AgentState:
    critique = state.get('critique', '')
    is_revision = bool(critique)
    if is_revision:
        print('Writer Agent: revising report based on critique...')
    else:
        print('Writer Agent: generating report...')
    try:
        critique_section = f'\n\nPrevious critique to address:\n{critique}' if is_revision else ''
        source_catalog, citation_rules = _format_source_catalog(state.get('search_results') or [])
        catalog_section = (
            f'\n\nRetrieved sources (map each [n] in the body to one APA entry under ## References):\n{source_catalog}\n'
            if source_catalog else '\n\nNo retrieved URLs — do not add a ## References section.\n'
        )
        mem = (state.get('memory_context') or '').strip()
        memory_section = ''
        if mem:
            memory_section = (
                '\n\nEarlier research from past runs in this app (for continuity only; not web sources — do not use [n] for these):\n'
                f'{mem}\n\n'
                'You may open the Executive Summary or Key Findings with a brief bridge when relevant '
                '(e.g. "Following up on our earlier look at EV adoption…") if it matches the memory; '
                'all factual claims tied to this run\'s web results must still use [n] from the catalog only.\n'
            )
        response = llm.invoke([
            SystemMessage(content=(
                'You are a professional report writer. Write a clear structured report with sections: '
                '# Title, ## Executive Summary, ## Key Findings, ## Detailed Analysis, ## Conclusion. '
                + citation_rules
            )),
            HumanMessage(
                content=(
                    f"Query: {state['query']}{memory_section}\n{catalog_section}\n"
                    f'Analysis (may contain [n] markers; keep them consistent with the catalog above):\n{state["analysis"]}'
                    f'{critique_section}\n\nWrite the full report now.'
                )
            ),
        ])
        print('   Report written')
        # Clear critique so the supervisor routes to critic again after a REVISE (not writer in a loop).
        return {
            **state,
            'report': response.content,
            'critique': '',
            'messages': state['messages'] + [AIMessage(content=f"Writer completed {'revision' if is_revision else 'report'} for: {state['query']}")]
        }
    except Exception as e:
        print(f'   Writer Agent failed: {e}')
        user_msg = user_message_for_groq_limit(e)
        return {
            **state,
            'report': f'Report generation failed: {user_msg}',
            'messages': state['messages'] + [AIMessage(content=f'Writer Agent failed: {user_msg}')],
        }
