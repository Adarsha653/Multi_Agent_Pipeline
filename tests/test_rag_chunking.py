from utils.rag_store import chunk_string


def test_chunk_string_empty():
    assert chunk_string('') == []
    assert chunk_string('   ') == []


def test_chunk_string_single_window():
    assert chunk_string('hello world', chunk_size=100, overlap=0) == ['hello world']


def test_chunk_string_overlap_produces_multiple():
    parts = chunk_string('abcdefghijklmnop', chunk_size=6, overlap=2)
    assert len(parts) >= 2
    joined = ''.join(parts)
    assert 'abc' in joined and 'mnop' in joined
