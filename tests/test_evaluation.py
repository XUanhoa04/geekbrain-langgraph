from geekbrain_rag.evaluation import valid_live_historical_comparison


def test_evaluator_accepts_observed_direction_when_live_jitter_crosses_baseline():
    question = "Compare PaymentGW's current p99 latency to its Q1 2026 daily average."
    expected = "Current ~185ms. Q1 daily average ~183ms. Current is slightly above."
    candidate = (
        "According to the live Monitoring API, current p99 latency is 178 ms; "
        "the Q1 2026 daily average from the analytics database is 183.0016 ms, "
        "so the observed value is below the historical average [3]."
    )
    assert valid_live_historical_comparison(question, candidate, expected)


def test_evaluator_rejects_inconsistent_or_unprovenanced_comparison():
    question = "Compare current p99 latency to its Q1 daily average."
    expected = "Current ~185ms. Q1 daily average ~183ms."
    wrong_direction = (
        "Live Monitoring API current p99 latency is 178 ms; Q1 daily average from the "
        "analytics database is 183 ms, so it is above [1]."
    )
    missing_provenance = "Current p99 latency is 178 ms; Q1 daily average is 183 ms, below [1]."
    assert not valid_live_historical_comparison(question, wrong_direction, expected)
    assert not valid_live_historical_comparison(question, missing_provenance, expected)
