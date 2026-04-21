from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from graph.state import AgentState
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal
from utils.groq_llm import chat_groq

load_dotenv()
llm = chat_groq()


class CriticVerdict(BaseModel):
    verdict: Literal['APPROVED', 'REVISE'] = Field(
        description='APPROVED if the report is acceptable; REVISE only if something critical is missing or wrong.'
    )
    feedback: str = Field(
        default='',
        description='If REVISE: concrete, actionable fixes for the writer. If APPROVED: may be empty.',
    )


def _fallback_critic_invoke(state: AgentState, system_content: str) -> tuple[bool, str]:
    """Plain-text fallback if structured parsing fails."""
    response = llm.invoke(
        [
            SystemMessage(content=system_content),
            HumanMessage(content=f"Query: {state['query']}\n\nReport:\n{state['report']}\n\nStart your reply with APPROVED or REVISE."),
        ]
    )
    text = response.content.strip()
    is_approved = text.upper().startswith('APPROVED')
    critique = text if not is_approved else ''
    return is_approved, critique


def critic_agent_node(state: AgentState) -> AgentState:
    print('Critic Agent: reviewing report...')
    has_web = bool(state.get('search_results'))
    citation_rule = (
        'Web results were provided for this run: the report MUST include a "## References" section with real APA 7th entries — not semicolon-separated catalog dumps. '
        'For **en.wikipedia.org** URLs, expect: *Wikipedia contributors.* (date). *Title.* *In Wikipedia.* Retrieved <date>, from <URL> — not `Wikipedia.` alone as author without *In Wikipedia.* '
        'Factual claims should use inline [n] markers tied to those sources. If references are malformed or missing while the body cites sources, REVISE. '
        if has_web
        else 'No web results were retrieved for this run: do not require ## References or [n] markers. '
    )
    system_content = (
        'You are a report reviewer. If the report has a title, summary, findings, and conclusion and is relevant to the query, '
        'verdict should be APPROVED. Use REVISE only if something critical is missing or factually wrong. Be lenient. '
        + citation_rule
    )

    revision_count = state.get('revision_count', 0) + 1
    try:
        structured_llm = llm.with_structured_output(CriticVerdict)
        parsed = structured_llm.invoke(
            [
                SystemMessage(content=system_content),
                HumanMessage(
                    content=f"Query: {state['query']}\n\nReport:\n{state['report']}\n\nReturn structured verdict and feedback."
                ),
            ]
        )
        if not isinstance(parsed, CriticVerdict):
            parsed = CriticVerdict.model_validate(parsed)
        is_approved = parsed.verdict == 'APPROVED'
        critique = '' if is_approved else (parsed.feedback.strip() or 'Please revise the report per reviewer notes.')
    except Exception as e:
        print(f'   Structured critic failed ({e}); using text fallback.')
        try:
            is_approved, critique = _fallback_critic_invoke(state, system_content)
        except Exception as e2:
            print(f'   Critic Agent failed: {e2}')
            return {
                **state,
                'is_approved': True,
                'critique': 'Critic failed, auto-approving',
                'revision_count': revision_count,
                'messages': state['messages'] + [AIMessage(content=f'Critic Agent failed: {e2}')],
            }

    if is_approved:
        print('   Report APPROVED')
    else:
        print(f'   Report needs REVISION (attempt {revision_count})')

    return {
        **state,
        'critique': critique,
        'is_approved': is_approved,
        'revision_count': revision_count,
        'messages': state['messages'] + [AIMessage(content=f"Critic verdict: {'APPROVED' if is_approved else 'REVISE'}")],
    }
