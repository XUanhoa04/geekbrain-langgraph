from __future__ import annotations

import re


def _metric_after(pattern: str, text: str) -> float | None:
    match = re.search(
        pattern + r"[^0-9]{0,80}([0-9][0-9,]*(?:\.[0-9]+)?)\s*ms",
        text,
        re.IGNORECASE,
    )
    return float(match.group(1).replace(",", "")) if match else None


def valid_live_historical_comparison(question: str, candidate: str, expected: str) -> bool:
    """Deterministically accept jittered live comparisons when their arithmetic is consistent."""
    if not (
        re.search(r"(?i)\bcurrent\b", question)
        and re.search(r"(?i)\b(?:daily\s+)?average\b", question)
        and re.search(r"(?i)\b(?:monitoring api|live api)\b", candidate)
        and re.search(r"(?i)\banalytics database\b", candidate)
    ):
        return False
    current = _metric_after(r"current(?:\s+p99)?(?:\s+latency)?", candidate)
    baseline = _metric_after(r"(?:q1\s+2026\s+)?daily\s+average", candidate)
    expected_current = _metric_after(r"current", expected)
    expected_baseline = _metric_after(r"(?:q1\s+)?daily\s+average", expected)
    if None in {current, baseline, expected_current, expected_baseline}:
        return False
    assert current is not None
    assert baseline is not None
    assert expected_current is not None
    assert expected_baseline is not None
    if abs(current - expected_current) > max(abs(expected_current) * 0.1, 1.0):
        return False
    if abs(baseline - expected_baseline) > max(abs(expected_baseline) * 0.02, 0.5):
        return False
    observed_direction = "above" if current > baseline else "below" if current < baseline else "equal"
    return bool(re.search(rf"(?i)\b{observed_direction}\b", candidate))
