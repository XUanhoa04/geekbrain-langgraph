from geekbrain_rag.agent import (
    _citation_ids,
    _cited_grounding_context,
    _has_usable_evidence,
    _valid_citations,
    answer_requirements,
    derive_capacity_evidence,
    derive_cross_source_evidence,
    derive_temporal_evidence,
    detect_intents,
    expand_database_queries,
    expand_document_queries,
    semantic_detect_intents,
    semantic_route,
)
from geekbrain_rag.config import Settings
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


def test_vietnamese_current_system_status_routes_live_metrics():
    assert detect_intents("Cho tôi xem trạng thái hệ thống hiện tại của AuthSvc") == [
        "LIVE_METRICS"
    ]


def test_semantic_router_handles_synonyms_without_lexical_trigger(monkeypatch):
    class FakeStructured:
        @staticmethod
        def invoke(_messages):
            return type(
                "Decision",
                (),
                {"intents": ["LIVE_METRICS"], "rationale": "present-time response speed"},
            )()

    class FakeModel:
        @staticmethod
        def with_structured_output(_schema):
            return FakeStructured()

    monkeypatch.setattr("geekbrain_rag.agent._llm", lambda _settings: FakeModel())
    assert semantic_detect_intents(
        "Tốc độ phản hồi của cổng thanh toán ngay lúc này ra sao?",
        Settings(),
        ("PaymentGW",),
    ) == ["LIVE_METRICS"]


def test_vietnamese_policy_and_current_latency_routes_both_sources():
    assert detect_intents("Chính sách SLA và độ trễ hiện tại của AuthSvc là gì?") == [
        "DOCUMENT",
        "DATABASE",
        "LIVE_METRICS",
    ]


def test_common_live_typo_and_database_size_do_not_fall_back_to_documents():
    assert detect_intents("How is the halth of payment gateway?") == ["LIVE_METRICS"]
    assert detect_intents("What is the db size?") == ["DATABASE"]


def test_non_live_status_word_does_not_trigger_fuzzy_live_route():
    assert detect_intents("What is the status of the deployment proposal?") == ["DOCUMENT"]


def test_holistic_database_expansion_is_bounded_and_multifaceted():
    queries = expand_database_queries(
        "Assess whether PaymentGW is reliable and recommend improvements.",
        ("PaymentGW", "InventorySvc"),
        ("HOLISTIC_SYNTHESIS", "RELIABILITY", "SLA_ASSESSMENT"),
    )
    assert len(queries) <= 5
    assert any("incident history" in query and "PaymentGW" in query for query in queries)
    assert any("root cause/type" in query for query in queries)
    assert any("operational metric aggregates" in query for query in queries)
    assert any("matching SLA targets" in query for query in queries)


def test_citations_must_exist_and_be_in_range():
    assert _valid_citations("Fact [1]. Another fact [2].", 2)
    assert not _valid_citations("Fact without support.", 2)
    assert not _valid_citations("Made-up source [9].", 2)


def test_grouped_and_adjacent_citations_are_supported_but_malformed_groups_fail():
    assert _citation_ids("Facts [1, 2] and [3][4].") == [1, 2, 3, 4]
    assert _valid_citations("Facts [1, 2] and [3][4].", 4)
    assert not _valid_citations("Broken [1, source].", 4)


def test_missing_citations_never_expand_grounding_to_all_evidence():
    evidence = [
        {"citation_id": 1, "source": "a", "content": "secretly unrelated"},
        {"citation_id": 2, "source": "b", "content": "also unrelated"},
    ]
    assert _cited_grounding_context("Unsupported answer.", evidence) == ""


def test_zero_or_error_only_evidence_forces_abstention_path():
    assert not _has_usable_evidence([])
    assert not _has_usable_evidence([{"kind": "DOCUMENT_ERROR"}])
    assert _has_usable_evidence([{"kind": "DATABASE"}, {"kind": "DOCUMENT_ERROR"}])


def test_current_api_reference_is_not_live_monitoring():
    assert detect_intents("What is the current API rate limit for PaymentGW?") == ["DOCUMENT"]


def test_quarterly_review_document_does_not_imply_database():
    question = "The Q1 2026 review mentioned NotificationSvc capacity planning concerns"
    assert detect_intents(question) == ["DOCUMENT"]
    assert any(
        "capacity planning" in item
        for item in expand_document_queries(question, tasks=("CAPACITY_RISK",))
    )


def test_answer_requirements_preserve_named_source_types():
    requirements = answer_requirements(
        "What should a new engineer know from onboarding and team info?",
        tasks=("ONBOARDING", "OWNERSHIP"),
    )
    assert "access/setup" in requirements
    assert "technology stack" in requirements


def test_answer_requirements_identify_live_monitoring_provenance():
    requirements = answer_requirements(
        "What is PaymentGW's current p99 latency?", tasks=("LIVE_STATUS",)
    )
    assert "live Monitoring API" in requirements


def test_vietnamese_semantic_tasks_drive_onboarding_expansion_and_contract():
    question = "Cho mình xin quy trình cho nhân sự mới"
    queries = expand_document_queries(question, tasks=("ONBOARDING",))
    requirements = answer_requirements(question, tasks=("ONBOARDING",))
    assert any("onboarding checklist access training" in query for query in queries)
    assert "required training" in requirements


def test_vietnamese_cost_task_expands_analytics_without_english_cost_keyword():
    queries = expand_database_queries(
        "Tình trạng chi phí quý 1",
        tasks=("COST_OPTIMIZATION",),
    )
    requirements = answer_requirements(
        "Tình trạng chi phí quý 1",
        tasks=("COST_OPTIMIZATION",),
    )
    assert any("cost totals and trends" in query for query in queries)
    assert "historical spending period and total" in requirements


def test_semantic_router_returns_language_independent_task_taxonomy(monkeypatch):
    class FakeStructured:
        @staticmethod
        def invoke(_messages):
            from geekbrain_rag.agent import RouteDecision

            return RouteDecision(
                intents=["DOCUMENT"],
                rationale="Vietnamese employee induction request",
                tasks=["ONBOARDING"],
                document_queries=["quy trình tiếp nhận và đào tạo nhân sự mới"],
            )

    class FakeModel:
        @staticmethod
        def with_structured_output(_schema):
            return FakeStructured()

    monkeypatch.setattr("geekbrain_rag.agent._llm", lambda _settings: FakeModel())
    decision = semantic_route("Cho mình xin quy trình cho nhân sự mới", Settings())
    assert decision.tasks == ["ONBOARDING"]
    assert decision.document_queries == ["quy trình tiếp nhận và đào tạo nhân sự mới"]


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


def test_cross_source_derivation_reports_unavailable_live_service():
    items = [
        Evidence(
            "DATABASE",
            '{"rows":[{"service":"InventorySvc","metric":"latency_p99_ms","target":200}]}',
            "db",
        ),
        Evidence(
            "LIVE_METRICS",
            '{"observed_services":{},"errors":{"InventorySvc":"connection refused"}}',
            "api",
        ),
    ]
    derived = derive_cross_source_evidence("Is InventorySvc healthy?", items)
    assert len(derived) == 1
    assert '"observation_status": "UNAVAILABLE"' in derived[0].content
    assert "not evidence of health" in derived[0].content


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


def test_deadline_derivation_accepts_iso_and_vietnamese_numeric_dates():
    for raw in ("Deadline: 2026-03-15", "Hạn chót: 15/03/2026"):
        derived = derive_temporal_evidence(
            "Việc này đã quá hạn chưa?", [Evidence("DOCUMENT", raw, "plan.md")]
        )
        assert len(derived) == 1
        assert '"deadline": "2026-03-15"' in derived[0].content


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
