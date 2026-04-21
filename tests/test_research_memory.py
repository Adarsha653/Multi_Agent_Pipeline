"""JSON research memory: format, record, trim."""

import json

import pytest

from utils import research_memory as rm


@pytest.fixture
def isolated_memory(monkeypatch, tmp_path):
    path = tmp_path / 'mem.json'
    monkeypatch.setenv('RESEARCH_MEMORY_PATH', str(path))
    monkeypatch.setenv('RESEARCH_MEMORY_MAX_ENTRIES', '4')
    yield path
    monkeypatch.delenv('RESEARCH_MEMORY_PATH', raising=False)


def test_format_empty(isolated_memory):
    assert rm.format_memory_for_prompt('anything') == ''


def test_record_and_format_ranking(isolated_memory):
    long_ev = (
        '## Executive Summary\n'
        'EV sales grew in 2024 across major European markets, with policy incentives and charging '
        'infrastructure expansion driving adoption in urban corridors and fleet operators.'
    )
    long_q = (
        '## Executive Summary\n'
        'Qubits and superposition remain central to introductory quantum computing explanations, '
        'with error correction and hardware platforms still maturing for practical workloads.'
    )
    rm.record_research_memory('electric vehicles in Europe', long_ev)
    rm.record_research_memory('quantum computing basics', long_q)
    block = rm.format_memory_for_prompt('What about EV charging networks in Europe?')
    assert 'electric vehicles' in block.lower()
    assert 'charging' in block.lower() or 'European' in block or 'european' in block.lower()


def test_max_entries_trim_oldest(isolated_memory):
    pad = 'word ' * 30
    for i in range(6):
        rm.record_research_memory(
            f'query-{i}',
            f'## Executive Summary\nThis is storage entry {i}. {pad}End.',
        )
    data = json.loads(isolated_memory.read_text(encoding='utf-8'))
    assert len(data['entries']) == 4
    assert data['entries'][0]['query'] == 'query-2'


def test_extract_summary_prefers_executive_section():
    r = '# T\n## Executive Summary\nAlpha beta.\n## Key Findings\nGamma.'
    assert 'Alpha beta' in rm.extract_report_summary(r)
