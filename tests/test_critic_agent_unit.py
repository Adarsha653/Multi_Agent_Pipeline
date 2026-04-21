"""Critic agent: mock structured LLM; assert verdict and revision_count."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import agents.critic_agent as critic_mod
from agents.critic_agent import CriticVerdict
from tests.conftest import state_after_analysis


def _state_with_report(**kw):
    s = state_after_analysis()
    s['report'] = '# Report\n## Executive Summary\nOk [1].\n## References\n1. Example.'
    s.update(kw)
    return s


@patch.object(critic_mod, 'llm')
def test_critic_approved(mock_llm):
    chain = MagicMock()
    chain.invoke.return_value = CriticVerdict(verdict='APPROVED', feedback='')
    mock_llm.with_structured_output.return_value = chain

    out = critic_mod.critic_agent_node(_state_with_report())

    assert out['is_approved'] is True
    assert out['critique'] == ''
    assert out['revision_count'] == 1
    assert out['agent_steps'][-1]['agent'] == 'critic_agent'


@patch.object(critic_mod, 'llm')
def test_critic_revise(mock_llm):
    chain = MagicMock()
    chain.invoke.return_value = CriticVerdict(verdict='REVISE', feedback='Expand section X.')
    mock_llm.with_structured_output.return_value = chain

    out = critic_mod.critic_agent_node(_state_with_report(revision_count=0))

    assert out['is_approved'] is False
    assert 'Expand' in out['critique']
    assert out['revision_count'] == 1
