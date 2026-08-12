from geekbrain_rag.presentation import remove_private_reasoning, source_items


def test_private_reasoning_is_never_rendered() -> None:
    raw = "<thinking>internal chain of thought</thinking>Grounded answer [1]."

    assert remove_private_reasoning(raw) == "Grounded answer [1]."


def test_source_items_follow_current_evidence_format_and_citations() -> None:
    evidence = """[1] kind=DOCUMENT; source=sla_policy.md; score=0.91
Payment services target 99.95% monthly availability.

---

[2] kind=DATABASE; source=GeekBrain analytics DB; score=n/a
The monthly cost was $12,400."""

    items = source_items([evidence, evidence], "Availability is 99.95% [1].")

    assert items == [
        ("[1] sla_policy.md", "Payment services target 99.95% monthly availability.")
    ]


def test_source_items_support_legacy_named_sources() -> None:
    evidence = """[Source: monitoring_api]
The service latency is 182 ms."""

    items = source_items([evidence], "The monitoring_api reports 182 ms latency.")

    assert items == [("monitoring_api", "The service latency is 182 ms.")]
