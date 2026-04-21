"""Fix #7 — deterministic supervisor routing (no LLM)."""

from agents.supervisor import supervisor_node


def _state(**overrides):
    base = {
        'query': 'test query',
        'messages': [],
        'document_ids': [],
        'search_ran': True,
        'search_results': [{'url': 'https://example.com', 'title': 't', 'content': 'c'}],
        'analysis': 'synthesized analysis',
        'report': '# Report\nbody',
        'critique': '',
        'is_approved': False,
        'next_agent': '',
        'revision_count': 0,
        'memory_context': '',
    }
    base.update(overrides)
    return base


def test_approved_routes_to_end():
    out = supervisor_node(_state(is_approved=True, critique='still here'))
    assert out['next_agent'] == 'END'


def test_revision_cap_routes_to_end():
    out = supervisor_node(_state(revision_count=2, is_approved=False, critique='revise'))
    assert out['next_agent'] == 'END'


def test_needs_search():
    out = supervisor_node(_state(search_results=[], search_ran=False))
    assert out['next_agent'] == 'search_agent'


def test_empty_search_results_still_routes_to_analysis_after_search_ran():
    out = supervisor_node(_state(search_results=[], search_ran=True, analysis=''))
    assert out['next_agent'] == 'analysis_agent'


def test_needs_analysis():
    out = supervisor_node(_state(analysis=''))
    assert out['next_agent'] == 'analysis_agent'


def test_needs_writer():
    out = supervisor_node(_state(report=''))
    assert out['next_agent'] == 'writer_agent'


def test_needs_critic_first_pass():
    out = supervisor_node(_state(report='# ok', critique=''))
    assert out['next_agent'] == 'critic_agent'


def test_revise_routes_back_to_writer():
    out = supervisor_node(
        _state(
            report='# ok',
            critique='Please add sources',
            is_approved=False,
            revision_count=1,
        )
    )
    assert out['next_agent'] == 'writer_agent'


def test_preserves_other_state_keys():
    s = _state()
    out = supervisor_node(s)
    assert out['query'] == s['query']
    assert out['report'] == s['report']


def test_pipeline_failure_report_skips_critic_and_ends():
    out = supervisor_node(
        _state(
            report='Report generation failed: rate limit',
            search_ran=True,
            search_results=[{'url': 'https://x', 'title': 't', 'content': 'c'}],
            analysis='done',
            critique='',
            is_approved=False,
        )
    )
    assert out['next_agent'] == 'END'
    assert out['is_approved'] is False
