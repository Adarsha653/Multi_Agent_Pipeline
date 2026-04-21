"""Fix #8 / #10 / #12 — optional API keys for research vs report routes."""

import os
import secrets

from fastapi import Header, HTTPException


def _extract_token(x_api_key: str | None, authorization: str | None) -> str:
    if x_api_key and x_api_key.strip():
        return x_api_key.strip()
    if authorization:
        a = authorization.strip()
        if a.lower().startswith('bearer '):
            return a[7:].strip()
    return ''


def _digest_match(expected: str, got: str) -> bool:
    if not expected or not got:
        return False
    if len(got) != len(expected):
        return False
    return secrets.compare_digest(got, expected)


def check_pipeline_api_key(x_api_key: str | None, authorization: str | None) -> None:
    """POST /research only: if PIPELINE_API_KEY is set, require a matching token."""
    expected = (os.getenv('PIPELINE_API_KEY') or '').strip()
    if not expected:
        return
    got = _extract_token(x_api_key, authorization)
    if not _digest_match(expected, got):
        raise HTTPException(status_code=401, detail='Invalid or missing API key')


def check_reports_api_access(x_api_key: str | None, authorization: str | None) -> None:
    """GET /reports*: if either key env is set, require a match (Fix #12 — read-only key optional)."""
    pipeline = (os.getenv('PIPELINE_API_KEY') or '').strip()
    read_only = (os.getenv('PIPELINE_REPORTS_API_KEY') or '').strip()
    if not pipeline and not read_only:
        return
    got = _extract_token(x_api_key, authorization)
    if _digest_match(pipeline, got) or _digest_match(read_only, got):
        return
    raise HTTPException(status_code=401, detail='Invalid or missing API key')


def pipeline_api_key_dependency(
    x_api_key: str | None = Header(default=None, alias='X-API-Key'),
    authorization: str | None = Header(default=None),
) -> None:
    check_pipeline_api_key(x_api_key, authorization)


def reports_api_key_dependency(
    x_api_key: str | None = Header(default=None, alias='X-API-Key'),
    authorization: str | None = Header(default=None),
) -> None:
    check_reports_api_access(x_api_key, authorization)
