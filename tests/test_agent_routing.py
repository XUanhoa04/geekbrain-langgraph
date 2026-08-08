from geekbrain_rag.agent import (
    _cited_grounding_context,
    _valid_citations,
    answer_requirements,
    derive_capacity_evidence,
    derive_cross_source_evidence,
    derive_holistic_evidence,
    derive_temporal_evidence,
    detect_intents,
    deterministic_computational_answer,
    expand_database_queries,
    expand_document_queries,
)
from geekbrain_rag.retrieval import Evidence


def test_multi_intent_current_metric_vs_sla():
    assert detect_intents("Is PaymentGW's current error rate within its SLA target?") == [
        "DATABASE",
        "LIVE_METRICS",
    ]


def test_current_sla_status_routes_live_and_database():
    assert detect_intents("Is NotificationSvc currently meeting its SLA targets?") == [
        "DATABASE",
        "LIVE_METRICS",
    ]


def test_follow_up_sla_without_word_target_routes_database_and_live():
    assert detect_intents("Is FraudDetector's current error rate within its SLA?") == [
        "DATABASE",
        "LIVE_METRICS",
    ]


def test_multi_clause_cost_and_cause():
    intents = detect_intents(
        "Which service had the highest cost in March, and what caused that increase?"
    )
    assert "DATABASE" in intents
    assert "DOCUMENT" in intents


def test_holistic_routes_all_sources():
    assert detect_intents(
        "Prepare a comprehensive reliability report card across all services"
    ) == [
        "DOCUMENT",
        "DATABASE",
        "LIVE_METRICS",
    ]


def test_investigation_language_routes_all_sources():
    assert detect_intents("Assess whether PaymentGW is reliable and recommend improvements.") == [
        "DOCUMENT",
        "DATABASE",
        "LIVE_METRICS",
    ]


def test_incident_and_growth_questions_route_database():
    assert "DATABASE" in detect_intents("Has NotificationSvc had any incidents recently?")
    assert "DATABASE" in detect_intents("How fast has AuthSvc traffic been growing?")


def test_current_request_volume_routes_live_metrics():
    assert detect_intents("What is AuthSvc's current request volume?") == ["LIVE_METRICS"]


def test_holistic_database_expansion_is_bounded_and_multifaceted():
    queries = expand_database_queries(
        "Assess whether PaymentGW is reliable and recommend improvements."
    )
    assert len(queries) == 5
    assert any("incident history for PaymentGW" in query for query in queries)
    assert any("monthly cost trend for PaymentGW" in query for query in queries)
    assert any("SLA targets" in query for query in queries)


def test_citations_must_exist_and_be_in_range():
    assert _valid_citations("Fact [1]. Another fact [2].", 2)
    assert not _valid_citations("Fact without support.", 2)
    assert not _valid_citations("Made-up source [9].", 2)


def test_current_api_reference_is_not_live_monitoring():
    assert detect_intents("What is the current API rate limit for PaymentGW?") == ["DOCUMENT"]


def test_quarterly_review_document_does_not_imply_database():
    question = "The Q1 2026 review mentioned NotificationSvc capacity planning concerns"
    assert detect_intents(question) == ["DOCUMENT"]
    assert any("capacity planning" in item for item in expand_document_queries(question))


def test_answer_requirements_preserve_named_source_types():
    requirements = answer_requirements("What should a new engineer know from onboarding and team info?")
    assert "access/setup" in requirements
    assert "technology stack" in requirements


def test_answer_requirements_identify_live_monitoring_provenance():
    requirements = answer_requirements("What is PaymentGW's current p99 latency?")
    assert "live Monitoring API" in requirements


def test_grounding_context_contains_only_cited_evidence():
    evidence = [
        {"citation_id": 1, "source": "a", "content": "used"},
        {"citation_id": 2, "source": "b", "content": "unused"},
    ]
    context = _cited_grounding_context("Claim [1].", evidence)
    assert "used" in context
    assert "unused" not in context


def test_cross_source_daily_average_derivation_is_deterministic():
    items = [
        Evidence(
            "DATABASE",
            '{"rows":[{"service":"PaymentGW","avg_latency_p99_ms":183.0}]}',
            "db",
        ),
        Evidence(
            "LIVE_METRICS",
            '{"observed_services":{"PaymentGW":{"latency_ms":{"p99":185}}}}',
            "api",
        ),
    ]
    derived = derive_cross_source_evidence("Compare PaymentGW latency", items)
    assert len(derived) == 1
    assert '"current_value": 185.0' in derived[0].content
    assert '"historical_average": 183.0' in derived[0].content
    assert '"difference": 2.0' in derived[0].content
    assert '"observed_direction": "above"' in derived[0].content


def test_cross_source_sla_derivation_includes_each_metric_verdict():
    items = [
        Evidence(
            "DATABASE",
            """{"rows":[
                {"service":"NotificationSvc","metric":"latency_p99_ms","target":2000},
                {"service":"NotificationSvc","metric":"error_rate_percent","target":1.0}
            ]}""",
            "db",
        ),
        Evidence(
            "LIVE_METRICS",
            """{"observed_services":{"NotificationSvc":{
                "latency_ms":{"p99":3200},"error_rate_percent":2.1
            }}}""",
            "api",
        ),
    ]
    derived = derive_cross_source_evidence("SLA targets", items)
    assert derived[0].content.count('"verdict": "BREACH"') == 2
    assert '"difference": 1200.0' in derived[0].content
    assert '"percentage_difference": 110.0' in derived[0].content


def test_cross_source_derivation_abstains_on_malformed_or_missing_source():
    items = [Evidence("DATABASE", "not-json", "db")]
    assert derive_cross_source_evidence("Compare", items) == []


def test_deadline_derivation_uses_governed_document_date():
    items = [
        Evidence(
            "DOCUMENT",
            "Circuit breaker review scheduled April 15, 2026. All engineers attend.",
            "postmortem.md",
        )
    ]
    derived = derive_temporal_evidence("Is the review overdue?", items)
    assert len(derived) == 1
    assert '"deadline": "2026-04-15"' in derived[0].content
    assert '"status": "OVERDUE"' in derived[0].content


def test_capacity_derivation_compares_live_value_to_document_threshold():
    items = [
        Evidence(
            "DOCUMENT",
            "AuthSvc is projected at 35,000 req/min by end of Q2.",
            "capacity.md",
        ),
        Evidence(
            "LIVE_METRICS",
            '{"observed_services":{"AuthSvc":{"requests_per_minute":28000}}}',
            "api",
        ),
    ]
    derived = derive_capacity_evidence("Is AuthSvc close to its capacity threshold?", items)
    assert len(derived) == 1
    assert '"planned_capacity_threshold": 35000.0' in derived[0].content
    assert '"capacity_utilization_percent": 80.0' in derived[0].content
    assert '"proximity_verdict": "CLOSE_TO_THRESHOLD"' in derived[0].content


def test_holistic_digest_is_extractive_and_keeps_lineage():
    items = [
        Evidence(
            "DATABASE",
            """{"purpose":"Q1 incident count","rows":[{"service":"PaymentGW",
            "incident_count":3,"worst_p_number":1,"total_duration_minutes":180}]}""",
            "db",
        ),
        Evidence(
            "DOCUMENT",
            "Recommendation: implement circuit breaker fallback routing.",
            "paymentgw-postmortem.md",
        ),
    ]
    derived = derive_holistic_evidence(
        "Assess whether PaymentGW is reliable and recommend improvements.", items
    )
    assert len(derived) == 1
    assert "implement circuit breaker fallback routing" in derived[0].content
    assert "PaymentGW had 3 Q1 incidents" in derived[0].content
    assert sorted(derived[0].metadata["lineage"]) == ["db", "paymentgw-postmortem.md"]


def test_growth_answer_is_rendered_from_database_derivation():
    evidence = [
        {
            "citation_id": 1,
            "kind": "DATABASE",
            "content": """{"derived":{"start_month":"2026-01",
            "start_average_requests_per_minute":24000,"end_month":"2026-03",
            "end_average_requests_per_minute":27000,
            "absolute_growth_requests_per_minute":3000,"percentage_growth":12.5}}""",
        }
    ]
    answer = deterministic_computational_answer("How fast is it growing?", evidence)
    assert answer is not None
    assert "12.5%" in answer
    assert "24000" in answer and "27000" in answer
