"""Fix #8 / #12 — optional API key gates."""

import pytest
from fastapi import HTTPException

from utils.api_auth import check_pipeline_api_key, check_reports_api_access


def _clear_report_env(monkeypatch):
    monkeypatch.delenv('PIPELINE_API_KEY', raising=False)
    monkeypatch.delenv('PIPELINE_REPORTS_API_KEY', raising=False)


def test_auth_disabled_when_env_unset(monkeypatch):
    monkeypatch.delenv('PIPELINE_API_KEY', raising=False)
    check_pipeline_api_key(None, None)


def test_valid_x_api_key(monkeypatch):
    monkeypatch.setenv('PIPELINE_API_KEY', 'my-secret-token')
    check_pipeline_api_key('my-secret-token', None)


def test_valid_bearer(monkeypatch):
    monkeypatch.setenv('PIPELINE_API_KEY', 'my-secret-token')
    check_pipeline_api_key(None, 'Bearer my-secret-token')


def test_missing_key_raises(monkeypatch):
    monkeypatch.setenv('PIPELINE_API_KEY', 'secret')
    with pytest.raises(HTTPException) as exc:
        check_pipeline_api_key(None, None)
    assert exc.value.status_code == 401


def test_wrong_key_raises(monkeypatch):
    monkeypatch.setenv('PIPELINE_API_KEY', 'secret')
    with pytest.raises(HTTPException) as exc:
        check_pipeline_api_key('wrong', None)
    assert exc.value.status_code == 401


def test_reports_access_disabled_when_no_env(monkeypatch):
    _clear_report_env(monkeypatch)
    check_reports_api_access(None, None)


def test_reports_access_accepts_read_key(monkeypatch):
    _clear_report_env(monkeypatch)
    monkeypatch.setenv('PIPELINE_REPORTS_API_KEY', 'read-only-secret')
    check_reports_api_access('read-only-secret', None)


def test_reports_access_accepts_pipeline_key_when_both(monkeypatch):
    _clear_report_env(monkeypatch)
    monkeypatch.setenv('PIPELINE_API_KEY', 'aaaaaaaaaaaaaaaa')
    monkeypatch.setenv('PIPELINE_REPORTS_API_KEY', 'bbbbbbbbbbbbbbbb')
    check_reports_api_access('aaaaaaaaaaaaaaaa', None)
    check_reports_api_access('bbbbbbbbbbbbbbbb', None)
