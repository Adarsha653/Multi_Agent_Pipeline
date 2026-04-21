"""Shared ChatGroq settings (Fix #6: timeouts + retries for all agents and API helpers)."""

import os

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
