import sqlite3

import pytest

from geekbrain_rag.analytics import (
    AnalyticsEngine,
    SQLPlan,
    deterministic_plan,
    validate_readonly_sql,
)


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM incidents",
        "SELECT * FROM incidents; DROP TABLE incidents",
        "SELECT * FROM incidents UNION SELECT * FROM monthly_costs",
        "PRAGMA table_info(incidents)",
        "SELECT * FROM sqlite_master",
        "SELECT load_extension('evil') FROM incidents",
    ],
)
def test_rejects_unsafe_sql(sql):
    with pytest.raises(ValueError):
        validate_readonly_sql(sql)


def test_accepts_parameterized_allowlisted_select():
    sql = validate_readonly_sql(
        "SELECT service, SUM(total_cost) AS cost FROM monthly_costs WHERE month BETWEEN ? AND ? GROUP BY service"
    )
    assert sql.startswith("SELECT")
    plan = SQLPlan(sql=sql, parameters=["2026-01", "2026-03"], purpose="Q1 cost")
    assert plan.sql.count("?") == len(plan.parameters)


def test_sqlite_query_only_is_available():
    with sqlite3.connect(":memory:") as conn:
        assert conn.execute("PRAGMA query_only").fetchone()[0] == 0
        conn.execute("PRAGMA query_only=ON")
        assert conn.execute("PRAGMA query_only").fetchone()[0] == 1


def test_deterministic_company_total_does_not_filter_geekbrain_as_service():
    plan = deterministic_plan(
        "What was GeekBrain's total infrastructure cost across all services in Q1 2026?"
    )
    assert plan is not None
    assert "service =" not in plan.sql
    assert plan.parameters == ["2026-01", "2026-03"]


def test_quarter_cost_derivation_is_deterministic():
    rows = [
        {"month": "2025-10", "total_cost": 100.0},
        {"month": "2025-11", "total_cost": 100.0},
        {"month": "2026-01", "total_cost": 300.0},
    ]
    derived = AnalyticsEngine.derive("cost from Q4 2025 to Q1 2026", rows)
    assert derived["absolute_change"] == 100.0
    assert derived["percentage_change"] == 50.0


def test_most_severe_incident_plan_orders_by_p_number():
    plan = deterministic_plan("Tell me about the most severe incident in Q1 2026.")
    assert plan is not None
    assert "SUBSTR(severity, 2)" in plan.sql
    assert plan.parameters == ["2026-01-01", "2026-03-31"]


def test_request_growth_derivation_is_deterministic():
    rows = [
        {"month": "2026-01", "avg_requests_per_minute": 24000},
        {"month": "2026-03", "avg_requests_per_minute": 27000},
    ]
    derived = AnalyticsEngine.derive("How fast is AuthSvc growing?", rows)
    assert derived["absolute_growth_requests_per_minute"] == 3000
    assert derived["percentage_growth"] == 12.5


def test_growth_projection_to_capacity_target_is_deterministic():
    rows = [
        {"month": "2026-01", "avg_requests_per_minute": 24000},
        {"month": "2026-03", "avg_requests_per_minute": 27000},
    ]
    derived = AnalyticsEngine.derive(
        "At that growth rate, when would AuthSvc hit 35k?", rows
    )
    assert derived["target_requests_per_minute"] == 35000
    assert 2 <= derived["estimated_quarters_to_target"] <= 3


def test_holistic_cost_summary_derives_reduction_target():
    rows = [
        {"service": "PaymentGW", "q1_total_cost": 16500},
        {"service": "FraudDetector", "q1_total_cost": 15900},
    ]
    derived = AnalyticsEngine.derive("Q1 2026 cost summary for all services", rows)
    assert derived["q1_total_cost"] == 32400
    assert derived["q2_reduction_target_15_percent"] == 4860
