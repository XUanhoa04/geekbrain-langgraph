from geekbrain_rag.monitoring import SERVICES, select_services


def test_explicit_service_is_bounded():
    assert select_services("What is PaymentGW current latency?") == ["PaymentGW"]


def test_ranking_fetches_all_services():
    assert select_services("Which service has the highest RPM?") == list(SERVICES)
