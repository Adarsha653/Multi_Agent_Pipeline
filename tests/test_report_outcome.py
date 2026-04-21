from utils.report_outcome import (
    is_pipeline_failure_report,
    saved_report_markdown_is_failure,
    skipped_eval_scores,
)


def test_detects_writer_failure_prefix():
    assert is_pipeline_failure_report('Report generation failed: 429') is True


def test_detects_analysis_failure_prefix():
    assert is_pipeline_failure_report('Analysis failed: timeout') is True


def test_normal_report_not_failure():
    assert is_pipeline_failure_report('# Title\n\n## Executive Summary\nHello.') is False


def test_skipped_eval_shape():
    s = skipped_eval_scores()
    assert s['overall'] is None
    assert 'skipped' in s['feedback'].lower() or 'did not finish' in s['feedback'].lower()


def test_saved_md_detects_query_plus_failure_body():
    raw = '''# Query\nWhat are the latest AI breakthroughs in 2025?\n\nReport generation failed: Error code: 429\n\n---\n**Approved:** True\n'''
    assert saved_report_markdown_is_failure(raw) is True


def test_saved_md_success_not_failure():
    raw = '''# Query\nNepal IT\n\n# Title\n## Executive Summary\nGrowth continued.\n'''
    assert saved_report_markdown_is_failure(raw) is False
