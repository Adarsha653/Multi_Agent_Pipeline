"""Shared ChatGroq settings (Fix #6: timeouts + retries for all agents and API helpers)."""

import os
import re

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

_DEFAULT_MODEL = 'llama-3.3-70b-versatile'


def _timeout_seconds() -> float:
    raw = os.getenv('GROQ_REQUEST_TIMEOUT', '120')
    try:
        return max(5.0, float(raw))
    except ValueError:
        return 120.0


def _max_retries() -> int:
    raw = os.getenv('GROQ_MAX_RETRIES', '2')
    try:
        return max(0, int(raw))
    except ValueError:
        return 2


def chat_groq(*, temperature: float = 0, model: str | None = None, **kwargs) -> ChatGroq:
    """Single place for Groq HTTP timeout and transport retries."""
    return ChatGroq(
        model=model or os.getenv('GROQ_MODEL', _DEFAULT_MODEL),
        temperature=temperature,
        timeout=_timeout_seconds(),
        max_retries=_max_retries(),
        **kwargs,
    )


def is_groq_rate_or_token_limit(exc: BaseException) -> bool:
    """True when Groq refused the call due to rate or daily token (TPD) limits."""
    s = str(exc).lower()
    if any(
        x in s
        for x in (
            '429',
            'rate limit',
            'rate_limit',
            'tokens per day',
            'token limit',
            'tpd',
            'quota',
            'rate_limit_exceeded',
        )
    ):
        return True
    return getattr(exc, 'status_code', None) == 429


_GROQ_COMPACT_DURATION = re.compile(
    r'^((?P<h>\d+)h)?((?P<m>\d+)m)?(?P<s>[\d.]+)s$',
    re.IGNORECASE,
)


def parse_groq_compact_duration_to_seconds(compact: str) -> int | None:
    """Parse Groq strings like `14m33.503999999s` or `1h30m0s` into total seconds (rounded)."""
    m = _GROQ_COMPACT_DURATION.match((compact or '').strip().lower())
    if not m:
        return None
    h = int(m.group('h') or 0)
    mi = int(m.group('m') or 0)
    try:
        sec_frac = float(m.group('s'))
    except ValueError:
        return None
    return int(round(h * 3600 + mi * 60 + sec_frac))


def format_duration_seconds_human(total_seconds: int) -> str:
    """Whole-number parts, e.g. `1 hr 2 mins 14 secs` (0 parts omitted except sub-minute)."""
    t = max(0, int(total_seconds))
    h = t // 3600
    rem = t % 3600
    m = rem // 60
    s = rem % 60
    parts: list[str] = []
    if h > 0:
        parts.append('1 hr' if h == 1 else f'{h} hrs')
    if m > 0:
        parts.append('1 min' if m == 1 else f'{m} mins')
    if s > 0:
        parts.append('1 sec' if s == 1 else f'{s} secs')
    if not parts:
        return '0 secs'
    return ' '.join(parts)


def humanize_groq_retry_hint(raw_hint: str | None) -> str | None:
    """Turn Groq's compact duration into a readable phrase; returns None if unknown shape."""
    if not raw_hint:
        return None
    stripped = raw_hint.strip()
    secs = parse_groq_compact_duration_to_seconds(stripped)
    if secs is not None:
        return format_duration_seconds_human(secs)
    return stripped


def extract_groq_retry_after_hint(exc: BaseException) -> str | None:
    """
    Pull a human-readable wait hint from Groq errors, e.g. 'Please try again in 20m24.288s'.
    Returns None if not present.
    """
    text = str(exc)
    # Typical TPD / RPM shape: 20m24.288s, 25m9s, 14m33.503999999s
    m = re.search(r'\btry\s+again\s+in\s+(\d+m\d+(?:\.\d+)?s)\b', text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r'\btry\s+again\s+in\s+(\d+(?:\.\d+)?\s*(?:seconds?|minutes?|hours?))\b', text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r'\btry\s+again\s+in\s+([^.\n\'\"]{3,48}?)(?:\.(?:\s|$|Need)|$)', text, re.IGNORECASE)
    if m:
        s = m.group(1).strip().rstrip('.,;: ')
        return s if len(s) >= 2 else None
    return None


def user_message_for_groq_limit(exc: BaseException) -> str:
    """
    Short message for HTTP responses or report bodies when Groq blocks the request.
    Falls back to str(exc) for unrelated errors.
    """
    if not is_groq_rate_or_token_limit(exc):
        return str(exc)
    base = (
        'Your Groq API usage has hit the current limit (for example daily tokens on the free '
        'or on-demand tier, or a short burst cap). Research cannot run until the limit resets '
        'or you increase your quota. Check usage and upgrade options at '
        'https://console.groq.com/settings/billing.'
    )
    hint = extract_groq_retry_after_hint(exc)
    if hint:
        friendly = humanize_groq_retry_hint(hint) or hint
        return (
            f'{base} Groq estimates you can retry after about {friendly} '
        )
    return (
        f'{base} If this is a short burst limit, wait a few minutes and retry; for daily token '
        'caps, open the Groq console to see when your usage window resets.'
    )
