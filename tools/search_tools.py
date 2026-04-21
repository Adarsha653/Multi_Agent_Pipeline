import os
import re
import time
import urllib.parse

import httpx
from ddgs import DDGS
from dotenv import load_dotenv

load_dotenv()

# Wikipedia requires a descriptive User-Agent (https://meta.wikimedia.org/wiki/User-Agent_policy).
_WIKI_UA = (
    'MultiAgentResearchPipeline/1.0 '
    '(https://github.com/; local research pipeline; Python/httpx)'
)


def _search_max_attempts() -> int:
    try:
        return max(1, int(os.getenv('WEB_SEARCH_MAX_ATTEMPTS', '3')))
    except ValueError:
        return 3


def _search_backoff_sec() -> float:
    try:
        return max(0.0, float(os.getenv('WEB_SEARCH_BACKOFF_SEC', '0.6')))
    except ValueError:
        return 0.6


def _sparse_threshold() -> int:
    """Minimum DDGS hits before we try an alternate query and/or Wikipedia."""
    try:
        return max(1, int(os.getenv('WEB_SEARCH_SPARSE_THRESHOLD', '3')))
    except ValueError:
        return 3


DEFAULT_SNIPPET_MAX = 800


def _trim_snippet(text: str | None, max_chars: int) -> str:
    if not text:
        return ''
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + '…'


def _dedupe_key(row: dict) -> str:
    url = (row.get('url') or '').strip().lower().rstrip('/')
    if url:
        return f'url:{url}'
    title = (row.get('title') or '').strip()[:160]
    body = (row.get('content') or '')[:120]
    return f'fallback:{title}|{body}'


def dedupe_and_trim_search_results(
    results: list[dict],
    max_snippet: int = DEFAULT_SNIPPET_MAX,
) -> list[dict]:
    """Drop duplicate URLs (first wins), trim long bodies for downstream LLM context."""
    seen: set[str] = set()
    out: list[dict] = []
    for r in results:
        key = _dedupe_key(r)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            'title': r.get('title'),
            'url': r.get('url'),
            'content': _trim_snippet(r.get('content'), max_snippet),
            'score': r.get('score', 1.0),
        })
    return out


def _merge_search_results(primary: list[dict], extra: list[dict]) -> list[dict]:
    """Append rows from `extra` that do not duplicate `primary` by URL / fingerprint."""
    seen: set[str] = {_dedupe_key(r) for r in primary}
    out = list(primary)
    for r in extra:
        key = _dedupe_key(r)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def merge_search_results(primary: list[dict], extra: list[dict]) -> list[dict]:
    """Public wrapper: `primary` rows keep their order; `extra` rows append if not duplicates."""
    return _merge_search_results(primary, extra)


def _alternate_ddgs_query(query: str) -> str | None:
    """Broader or rephrased query for a second DDGS pass when the first is thin."""
    q = (query or '').strip()
    if not q:
        return None
    low = q.lower()
    words = q.split()
    if len(words) > 6:
        short = ' '.join(words[:6])
        if short.lower() != low:
            return short
    for suf in (' overview', ' explained', ' basics'):
        cand = (q + suf).strip()
        if cand.lower() != low:
            return cand
    stripped = re.sub(r'[\s?.!]+$', '', q)
    if stripped and stripped.lower() != low:
        return stripped
    return None


def _records_from_wikipedia_opensearch(data: object) -> list[dict]:
    """Parse Wikipedia `action=opensearch` JSON → same shape as DDGS rows."""
    if not isinstance(data, list) or len(data) < 4:
        return []
    titles, descs, urls = data[1], data[2], data[3]
    if not isinstance(titles, list) or not isinstance(urls, list):
        return []
    if not isinstance(descs, list):
        descs = []
    out: list[dict] = []
    for i, title in enumerate(titles):
        if i >= len(urls) or not isinstance(title, str):
            continue
        url = urls[i]
        if not isinstance(url, str) or not url.strip():
            continue
        desc = descs[i] if i < len(descs) and isinstance(descs[i], str) else ''
        out.append({
            'title': title,
            'url': url.strip(),
            'content': desc.strip() or title,
            'score': 0.95,
        })
    return out


def wikipedia_opensearch(query: str, limit: int = 5) -> list[dict]:
    """Secondary source: Wikipedia title search (opensearch API)."""
    q = (query or '').strip()
    if not q:
        return []
    params = urllib.parse.urlencode({
        'action': 'opensearch',
        'search': q,
        'limit': str(max(1, limit)),
        'namespace': '0',
        'format': 'json',
    })
    url = f'https://en.wikipedia.org/w/api.php?{params}'
    try:
        with httpx.Client(timeout=15.0) as client:
            r = client.get(url, headers={'User-Agent': _WIKI_UA})
            r.raise_for_status()
            return _records_from_wikipedia_opensearch(r.json())
    except Exception:
        return []


def _ddgs_text_search(query: str, max_results: int = 5) -> list[dict]:
    """Single DuckDuckGo text search with exponential backoff retries (Fix #6)."""
    max_attempts = _search_max_attempts()
    backoff = _search_backoff_sec()
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            results: list[dict] = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results, region='wt-en'):
                    results.append({
                        'title': r.get('title'),
                        'url': r.get('href'),
                        'content': r.get('body'),
                        'score': 1.0,
                    })
            return results
        except Exception as e:
            last_error = e
            if attempt < max_attempts - 1:
                time.sleep(backoff * (2**attempt))
    assert last_error is not None
    raise last_error


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """
    DuckDuckGo text search, then if fewer than WEB_SEARCH_SPARSE_THRESHOLD (default 3) hits:
    retry DDGS with a broader/alternate query, then merge Wikipedia opensearch results.
    """
    threshold = _sparse_threshold()
    primary = _ddgs_text_search(query, max_results)
    merged = list(primary)

    if len(merged) < threshold:
        alt = _alternate_ddgs_query(query)
        if alt:
            try:
                merged = _merge_search_results(merged, _ddgs_text_search(alt, max_results))
            except Exception:
                pass

    if len(merged) < threshold:
        wiki_limit = max(threshold, max_results, 5)
        merged = _merge_search_results(merged, wikipedia_opensearch(query, limit=wiki_limit))

    return merged
