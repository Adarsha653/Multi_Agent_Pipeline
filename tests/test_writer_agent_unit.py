"""Writer agent: mock LLM; assert report and cleared critique on success."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import agents.writer_agent as writer_mod
from tests.conftest import state_after_analysis


@patch.object(writer_mod, 'llm')
def test_writer_agent_produces_report(mock_llm):
    mock_llm.invoke.return_value = MagicMock(
        content='# Title\n## Executive Summary\nShort [1].\n## Key Findings\n- A [1]\n'
    )
    state = state_after_analysis()

    out = writer_mod.writer_agent_node(state)

    assert '# Title' in out['report']
    assert out['critique'] == ''
    assert out['agent_steps'][-1]['agent'] == 'writer_agent'


@patch.object(writer_mod, 'llm')
def test_writer_revision_keeps_critique_handling(mock_llm):
    mock_llm.invoke.return_value = MagicMock(content='# Title\nRevised body [1].')
    state = state_after_analysis()
    state['critique'] = 'Add more on topic B.'

    out = writer_mod.writer_agent_node(state)

    assert 'Revised' in out['report'] or 'Title' in out['report']
    assert out['critique'] == ''
