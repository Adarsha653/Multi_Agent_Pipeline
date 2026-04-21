"""Analysis agent: mock LLM; assert analysis and timing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import agents.analysis_agent as analysis_mod
from tests.conftest import state_after_search


@patch.object(analysis_mod, 'llm')
def test_analysis_agent_sets_analysis_and_step(mock_llm):
    mock_llm.invoke.return_value = MagicMock(
        content='## Findings\nPoint [1].\n\n## Source support overview\n**Strong (multiple sources):** [1].'
    )
    state = state_after_search()

    out = analysis_mod.analysis_agent_node(state)

    assert 'Source support overview' in out['analysis'] or 'Findings' in out['analysis']
    assert out['agent_steps'][-1]['agent'] == 'analysis_agent'
    assert len(out['messages']) > len(state['messages'])


@patch.object(analysis_mod, 'llm')
def test_analysis_agent_handles_empty_results(mock_llm):
    mock_llm.invoke.return_value = MagicMock(content='No sources; limited evidence.')
    from tests.conftest import minimal_agent_state

    state = minimal_agent_state(search_ran=True, search_results=[])

    out = analysis_mod.analysis_agent_node(state)

    assert 'analysis' in out
    assert out['agent_steps'][-1]['agent'] == 'analysis_agent'
