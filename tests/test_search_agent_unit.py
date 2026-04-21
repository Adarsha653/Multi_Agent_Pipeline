"""Search agent: mock LLM + web_search; assert state transitions."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

import agents.search_agent as search_mod
from tests.conftest import minimal_agent_state


@pytest.fixture
def state() -> dict:
    return minimal_agent_state(query='renewable energy trends')


@patch.object(search_mod, 'web_search')
@patch.object(search_mod, 'llm')
def test_search_agent_populates_results_and_marks_ran(mock_llm, mock_web, state):
    mock_llm.invoke.return_value = MagicMock(
        content='1. overview of renewable energy\n2. renewable energy 2025 news\n3. renewable energy challenges'
    )
    mock_web.return_value = [
        {'title': 'Hit', 'url': 'https://example.com/x', 'content': 'body text', 'score': 1.0},
    ]

    out = search_mod.search_agent_node(state)

    assert out['search_ran'] is True
    assert len(out['search_results']) >= 1
    assert out['search_results'][0]['url'] == 'https://example.com/x'
    assert out['agent_steps'][-1]['agent'] == 'search_agent'
    assert out['agent_steps'][-1]['seconds'] >= 0
    assert any('Search Agent retrieved' in (m.content or '') for m in out['messages'] if isinstance(m, AIMessage))


@patch.object(search_mod, 'web_search')
@patch.object(search_mod, 'llm')
def test_search_agent_failure_still_sets_search_ran(mock_llm, mock_web, state):
    mock_llm.invoke.side_effect = RuntimeError('LLM down')

    out = search_mod.search_agent_node(state)

    assert out['search_ran'] is True
    assert out['search_results'] == []
    assert out['agent_steps'][-1]['agent'] == 'search_agent'
