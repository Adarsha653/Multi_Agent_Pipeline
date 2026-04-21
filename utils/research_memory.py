"""Simple JSON-backed memory of past research runs (query + summary) for cross-session context."""

from __future__ import annotations

import json

from utils.report_outcome import is_pipeline_failure_report
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_STOP = frozenset({
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can',
    'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'between', 'under', 'again', 'further', 'then', 'once',
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
    'very', 'just', 'and', 'but', 'if', 'or', 'because', 'what', 'which', 'who', 'this', 'that',
    'these', 'those', 'it', 'its', 'about', 'against', 'any', 'our', 'your', 'their', 'them', 'his',
    'her', 'she', 'he', 'we', 'they', 'i', 'me', 'my', 'us', 'was', 'were',
})


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _memory_file() -> Path:
    env = (os.getenv('RESEARCH_MEMORY_PATH') or '').strip()
    if env:
        return Path(env)
    return _project_root() / 'data' / 'research_memory.json'


def _max_entries() -> int:
    try:
        return max(3, int(os.getenv('RESEARCH_MEMORY_MAX_ENTRIES', '25')))
    except ValueError:
        return 25


def _max_items_prompt() -> int:
    try:
        return max(1, min(12, int(os.getenv('RESEARCH_MEMORY_PROMPT_ITEMS', '5'))))
    except ValueError:
        return 5


def _tokens(text: str) -> set[str]:
    return {
        t for t in re.sub(r'[^a-z0-9\s]', ' ', (text or '').lower()).split()
        if len(t) > 1 and t not in _STOP
    }


def _load_raw() -> dict[str, Any]:
    path = _memory_file()
    if not path.exists():
        return {'entries': []}
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {'entries': []}
        entries = data.get('entries')
        if not isinstance(entries, list):
            return {'entries': []}
        return {'entries': entries}
    except (OSError, json.JSONDecodeError):
        return {'entries': []}


def _save_raw(data: dict[str, Any]) -> None:
    path = _memory_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.json.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def extract_report_summary(report: str) -> str:
    """Pull executive summary or a short lead-in for storage."""
    if not report or report.strip().startswith('Report generation failed'):
        return ''
    m = re.search(
        r'##\s*Executive\s*Summary\s*\n+(.+?)(?=\n##\s|\Z)',
        report,
        re.IGNORECASE | re.DOTALL,
    )
    if m:
        body = m.group(1).strip()
        return body[:3000] if len(body) > 3000 else body
    body = report.strip()
    return body[:1200] if len(body) > 1200 else body


def format_memory_for_prompt(query: str) -> str:
    """
    Build a short block for the LLM: ranked by token overlap with `query`, then recency.
    Empty string if no memory.
    """
    data = _load_raw()
    entries = [e for e in data.get('entries') or [] if isinstance(e, dict)]
    if not entries:
        return ''
    qtok = _tokens(query)
    ranked = sorted(
        entries,
        key=lambda e: (
            len(qtok & _tokens(str(e.get('query') or ''))),
            str(e.get('saved_at') or ''),
        ),
        reverse=True,
    )
    picked = ranked[:_max_items_prompt()]
    lines: list[str] = []
    for e in picked:
        q_e = str(e.get('query') or '').strip()
        s = str(e.get('summary') or '').strip().replace('\n', ' ')
        if len(s) > 650:
            s = s[:647] + '…'
        if q_e:
            lines.append(f'- Earlier topic: {q_e}\n  What we found then: {s}')
    return '\n'.join(lines)


def record_research_memory(query: str, report: str) -> None:
    """Append one session after a successful report (trim oldest past max)."""
    if is_pipeline_failure_report(report):
        return
    summary = extract_report_summary(report)
    if not summary or len(summary) < 80:
        return
    entry = {
        'query': query.strip()[:500],
        'summary': summary[:4000],
        'saved_at': datetime.now(timezone.utc).isoformat(),
    }
    data = _load_raw()
    entries: list[Any] = data['entries']
    entries.append(entry)
    cap = _max_entries()
    if len(entries) > cap:
        del entries[0 : len(entries) - cap]
    _save_raw({'entries': entries})
