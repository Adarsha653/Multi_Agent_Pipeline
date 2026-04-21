"""Evaluator and Groq helpers handle rate / token limits."""

from eval.evaluator import _scores_when_quota_exceeded
from utils.groq_llm import (
    extract_groq_retry_after_hint,
    format_duration_seconds_human,
    humanize_groq_retry_hint,
    is_groq_rate_or_token_limit,
    parse_groq_compact_duration_to_seconds,
    user_message_for_groq_limit,
)


def test_is_rate_limit_groq_message():
    err = RuntimeError(
        "Error code: 429 - {'error': {'message': 'Rate limit reached for model', 'code': 'rate_limit_exceeded'}}"
    )
    assert is_groq_rate_or_token_limit(err) is True


def test_is_rate_limit_tpd_message():
    err = RuntimeError('tokens per day (TPD): Limit 100000')
    assert is_groq_rate_or_token_limit(err) is True


def test_is_rate_limit_status_code():
    e = type('E', (), {'status_code': 429})()
    assert is_groq_rate_or_token_limit(e) is True


def test_not_rate_limit():
    assert is_groq_rate_or_token_limit(ValueError('invalid json')) is False


def test_quota_scores_null_dimensions():
    s = _scores_when_quota_exceeded()
    assert s['relevance'] is None
    assert s['overall'] is None
    assert 'rate limit' in s['feedback'].lower()


def test_user_message_friendly_for_limit():
    err = RuntimeError('429 rate_limit_exceeded')
    msg = user_message_for_groq_limit(err)
    assert 'Groq' in msg
    assert 'console.groq.com' in msg
    assert '429' not in msg or 'limit' in msg.lower()


def test_user_message_passthrough_other():
    err = ValueError('something else')
    assert user_message_for_groq_limit(err) == 'something else'


def test_extract_retry_hint_minutes_seconds():
    err = RuntimeError(
        "429 ... Please try again in 20m24.288s. Need more tokens"
    )
    assert extract_groq_retry_after_hint(err) == '20m24.288s'


def test_user_message_includes_humanized_retry():
    err = RuntimeError(
        "Error code: 429 - {'message': '... Please try again in 24m33.984s. ...', 'code': 'rate_limit_exceeded'}"
    )
    msg = user_message_for_groq_limit(err)
    assert '24 mins' in msg and '34 secs' in msg
    assert 'Groq estimates' in msg


def test_humanize_strips_float_noise():
    assert humanize_groq_retry_hint('14m33.503999999s') == '14 mins 34 secs'


def test_parse_hours_minutes_seconds():
    assert parse_groq_compact_duration_to_seconds('1h2m5s') == 3725
    assert format_duration_seconds_human(3725) == '1 hr 2 mins 5 secs'
