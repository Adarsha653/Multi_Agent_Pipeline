"""Fix #11 — /health and /ready probes."""

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_ok(client):
    r = client.get('/health')
    assert r.status_code == 200
    data = r.json()
    assert data['status'] == 'ok'
    assert 'service' in data


def test_ready_without_groq_key(client, monkeypatch):
    monkeypatch.delenv('GROQ_API_KEY', raising=False)
    r = client.get('/ready')
    assert r.status_code == 503
    assert 'GROQ' in r.json()['detail']


def test_ready_with_groq_key(client, monkeypatch):
    monkeypatch.setenv('GROQ_API_KEY', 'test-key-not-used-for-network')
    r = client.get('/ready')
    assert r.status_code == 200
    assert r.json()['status'] == 'ready'
