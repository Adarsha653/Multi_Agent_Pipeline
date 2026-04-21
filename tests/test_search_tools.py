"""Fix #7 — search result dedupe and snippet trimming; sparse-search fallbacks."""

from tools import search_tools as st
from tools.search_tools import dedupe_and_trim_search_results


def test_dedupe_same_url_keeps_first():
    rows = [
        {'title': 'First', 'url': 'https://x.com/a', 'content': 'one', 'score': 1.0},
        {'title': 'Dup', 'url': 'https://x.com/a/', 'content': 'two', 'score': 1.0},
    ]
    out = dedupe_and_trim_search_results(rows)
    assert len(out) == 1
    assert out[0]['title'] == 'First'
    assert out[0]['content'] == 'one'


def test_trims_long_snippet():
    long = 'word ' * 500
    rows = [{'title': 'T', 'url': 'https://z', 'content': long, 'score': 1.0}]
    out = dedupe_and_trim_search_results(rows, max_snippet=100)
    assert len(out) == 1
    assert len(out[0]['content']) <= 100
    assert out[0]['content'].endswith('…')


def test_empty_input():
    assert dedupe_and_trim_search_results([]) == []


def test_merge_search_results_dedupes_by_url():
    a = [{'title': 'A', 'url': 'https://x.com/1', 'content': 'c', 'score': 1.0}]
    b = [
        {'title': 'Dup', 'url': 'https://x.com/1/', 'content': 'other', 'score': 1.0},
        {'title': 'B', 'url': 'https://y.com/z', 'content': 'new', 'score': 1.0},
    ]
    out = st._merge_search_results(a, b)
    assert len(out) == 2
    assert out[0]['title'] == 'A'
    assert out[1]['title'] == 'B'


def test_wikipedia_opensearch_parse():
    payload = [
        'q',
        ['Photosynthesis', 'C4 carbon fixation'],
        ['Process used by plants…', 'Pathway'],
        [
            'https://en.wikipedia.org/wiki/Photosynthesis',
            'https://en.wikipedia.org/wiki/C4_carbon_fixation',
        ],
    ]
    rows = st._records_from_wikipedia_opensearch(payload)
    assert len(rows) == 2
    assert rows[0]['title'] == 'Photosynthesis'
    assert rows[0]['url'].endswith('Photosynthesis')
    assert 'plants' in rows[0]['content']
    assert rows[0]['score'] == 0.95


def test_alternate_query_truncates_long_phrase():
    q = 'one two three four five six seven eight'
    alt = st._alternate_ddgs_query(q)
    assert alt is not None
    assert alt.lower() != q.lower()
    assert len(alt.split()) == 6


def test_alternate_query_none_for_empty():
    assert st._alternate_ddgs_query('') is None
    assert st._alternate_ddgs_query('   ') is None
