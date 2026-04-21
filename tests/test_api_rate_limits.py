"""HTTP rate limits (slowapi) protect expensive routes and public demos."""

import pytest
from fastapi.testclient import TestClient

from api.main import app, read_reports_rate_limit, research_rate_limit


@pytest.fixture
def client():
    return TestClient(app)


def test_research_rate_limit_explicit_overrides_space(monkeypatch):
    monkeypatch.setenv('RESEARCH_RATE_LIMIT', '1/minute')
    monkeypatch.setenv('SPACE_ID', 'author/demo-space')
    assert research_rate_limit() == '1/minute'


def test_research_rate_limit_space_default(monkeypatch):
    monkeypatch.delenv('RESEARCH_RATE_LIMIT', raising=False)
    monkeypatch.setenv('SPACE_ID', 'author/demo-space')
    monkeypatch.delenv('SPACE_DEFAULT_RESEARCH_RATE_LIMIT', raising=False)
    assert research_rate_limit() == '5/minute'


def test_research_rate_limit_space_custom_default(monkeypatch):
    monkeypatch.delenv('RESEARCH_RATE_LIMIT', raising=False)
    monkeypatch.setenv('SPACE_ID', 'author/demo-space')
    monkeypatch.setenv('SPACE_DEFAULT_RESEARCH_RATE_LIMIT', '3/minute')
    assert research_rate_limit() == '3/minute'


def test_read_reports_rate_limit_space_default(monkeypatch):
    monkeypatch.delenv('READ_REPORTS_RATE_LIMIT', raising=False)
    monkeypatch.setenv('SPACE_ID', 'author/demo-space')
    monkeypatch.delenv('SPACE_DEFAULT_READ_REPORTS_RATE_LIMIT', raising=False)
    assert read_reports_rate_limit() == '60/minute'


def test_post_research_returns_429_when_over_limit(client, monkeypatch):
    """Cheap 400s still count toward the limit (abuse pattern)."""
    monkeypatch.delenv('PIPELINE_API_KEY', raising=False)
    monkeypatch.setenv('RESEARCH_RATE_LIMIT', '2/second')
    for _ in range(2):
        r = client.post('/research', json={'query': ''})
        assert r.status_code == 400
    over = client.post('/research', json={'query': ''})
    assert over.status_code == 429
    assert 'error' in over.json()


def test_get_reports_returns_429_when_over_limit(client, monkeypatch):
    monkeypatch.delenv('PIPELINE_API_KEY', raising=False)
    monkeypatch.delenv('PIPELINE_REPORTS_API_KEY', raising=False)
    monkeypatch.setenv('READ_REPORTS_RATE_LIMIT', '2/second')
    for _ in range(2):
        assert client.get('/reports').status_code == 200
    assert client.get('/reports').status_code == 429
