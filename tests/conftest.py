"""Shared fixtures for agent tests (full AgentState-shaped dicts)."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage


def minimal_agent_state(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        'query': 'sample research query',
        'search_ran': False,
        'messages': [],
        'search_results': [],
        'analysis': '',
        'report': '',
        'critique': '',
        'is_approved': False,
        'next_agent': '',
        'revision_count': 0,
        'memory_context': '',
        'report_format': 'markdown',
        'agent_steps': [],
    }
    base.update(overrides)
    return base


def state_after_search() -> dict[str, Any]:
    return minimal_agent_state(
        search_ran=True,
        search_results=[
            {'title': 'Example', 'url': 'https://example.com/a', 'content': 'Snippet A about topic.', 'score': 1.0},
            {'title': 'Other', 'url': 'https://example.org/b', 'content': 'Snippet B with more detail.', 'score': 1.0},
        ],
        messages=[AIMessage(content='search done')],
    )


def state_after_analysis() -> dict[str, Any]:
    s = state_after_search()
    s['analysis'] = 'Key finding [1]. Another [2].\n\n## Source support overview\n**Strong:** theme [1][2].'
    s['messages'] = list(s['messages']) + [AIMessage(content='analysis done')]
    return s
