from geekbrain_rag.presentation import (
    relevant_snippet,
    remove_private_reasoning,
    source_items,
)


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


def test_relevant_snippet_boundary_ellipsis() -> None:
    chunk = (
        "Header Section\n"
        "First line of content\n"
        "Second line describing payment metrics\n"
        "Third line with details\n"
        "Fourth line concluding remarks\n"
        "Final footer notes"
    )

    # Match near start: no leading ellipsis, trailing ellipsis present
    snippet_start = relevant_snippet(chunk, "Header Section and first line")
    assert not snippet_start.startswith("... ")
    assert snippet_start.endswith(" ...")

    # Match near end: leading ellipsis present, no trailing ellipsis
    snippet_end = relevant_snippet(chunk, "Final footer notes")
    assert snippet_end.startswith("... ")
    assert not snippet_end.endswith(" ...")

    # Match in middle: both leading and trailing ellipsis
    snippet_mid = relevant_snippet(chunk, "payment metrics with details")
    assert snippet_mid.startswith("... ")
    assert snippet_mid.endswith(" ...")
