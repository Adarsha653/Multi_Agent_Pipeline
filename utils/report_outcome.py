"""Detect reports that are pipeline error placeholders (not real deliverables)."""


def is_pipeline_failure_report(report: str | None) -> bool:
    """True for the writer/analysis error body stored in graph state (may include friendly Groq text)."""
    raw = (report or '').strip()
    if not raw:
        return False
    low = raw.lower()
    return low.startswith('report generation failed:') or low.startswith('analysis failed:')


def saved_report_markdown_is_failure(file_head: str) -> bool:
    """
    True for on-disk .md from _persist_report that embed a failed run (query header + error body).
    Used to hide legacy files from GET /reports and to avoid treating them as real deliverables.
    """
    low = (file_head or '').lower()
    if 'report generation failed:' in low:
        return True
    if 'analysis failed:' in low[:6000]:
        return True
    return False


def skipped_eval_scores() -> dict:
    """Placeholder scores when we do not call the evaluator LLM."""
    return {
        'relevance': None,
        'completeness': None,
        'clarity': None,
        'structure': None,
        'overall': None,
        'feedback': 'Scoring was skipped because the report did not finish successfully.',
    }
