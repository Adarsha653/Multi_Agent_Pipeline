"""Fix #10 / #12 — report routes honor API keys when set."""

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def _clear_keys(monkeypatch):
    monkeypatch.delenv('PIPELINE_API_KEY', raising=False)
    monkeypatch.delenv('PIPELINE_REPORTS_API_KEY', raising=False)


def test_root_never_requires_api_key(client, monkeypatch):
    monkeypatch.setenv('PIPELINE_API_KEY', 'super-secret-reports-key')
    r = client.get('/')
    assert r.status_code == 200


def test_reports_open_when_key_unset(client, monkeypatch):
    _clear_keys(monkeypatch)
    r = client.get('/reports')
    assert r.status_code == 200
    assert 'reports' in r.json()


def test_reports_require_key_when_set(client, monkeypatch):
    _clear_keys(monkeypatch)
    monkeypatch.setenv('PIPELINE_API_KEY', 'only-for-tests')
    assert client.get('/reports').status_code == 401
    ok = client.get('/reports', headers={'X-API-Key': 'only-for-tests'})
    assert ok.status_code == 200


def test_report_detail_requires_key_when_set(client, monkeypatch):
    _clear_keys(monkeypatch)
    monkeypatch.setenv('PIPELINE_API_KEY', 'only-for-tests')
    assert client.get('/reports/missing.md').status_code == 401
    r = client.get('/reports/missing.md', headers={'X-API-Key': 'only-for-tests'})
    assert r.status_code == 404


def test_reports_accept_read_only_key(client, monkeypatch):
    """Fix #12 — only PIPELINE_REPORTS_API_KEY protects reports."""
    _clear_keys(monkeypatch)
    monkeypatch.setenv('PIPELINE_REPORTS_API_KEY', 'read-only-secret')
    assert client.get('/reports').status_code == 401
    ok = client.get('/reports', headers={'X-API-Key': 'read-only-secret'})
    assert ok.status_code == 200


def test_reports_accept_either_key_when_both_set(client, monkeypatch):
    _clear_keys(monkeypatch)
    # Same length so secrets.compare_digest path is exercised
    monkeypatch.setenv('PIPELINE_API_KEY', 'aaaaaaaaaaaaaaaa')
    monkeypatch.setenv('PIPELINE_REPORTS_API_KEY', 'bbbbbbbbbbbbbbbb')
    assert client.get('/reports', headers={'X-API-Key': 'bbbbbbbbbbbbbbbb'}).status_code == 200
    assert client.get('/reports', headers={'X-API-Key': 'aaaaaaaaaaaaaaaa'}).status_code == 200


def test_research_rejects_read_only_key(client, monkeypatch):
    """Research never accepts PIPELINE_REPORTS_API_KEY alone."""
    _clear_keys(monkeypatch)
    monkeypatch.setenv('PIPELINE_API_KEY', 'aaaaaaaaaaaaaaaa')
    monkeypatch.setenv('PIPELINE_REPORTS_API_KEY', 'bbbbbbbbbbbbbbbb')
    r = client.post('/research', json={'query': 'x'}, headers={'X-API-Key': 'bbbbbbbbbbbbbbbb'})
    assert r.status_code == 401
