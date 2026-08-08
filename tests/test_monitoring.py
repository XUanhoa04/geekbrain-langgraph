from geekbrain_rag.monitoring import MAX_SERVICES_PER_QUERY, select_services

CATALOG = ("PaymentGW", "AuthSvc", "InventorySvc")


def test_explicit_service_is_bounded():
    assert select_services("What is PaymentGW current latency?", CATALOG) == ["PaymentGW"]


def test_ranking_fetches_all_services():
    assert select_services("Which service has the highest RPM?", CATALOG) == list(CATALOG)


def test_new_conventionally_named_service_is_not_lost():
    assert select_services("UserService hiện tại có khỏe không?", CATALOG) == ["UserService"]


def test_adversarial_service_list_is_bounded():
    services = " ".join(f"User{index}Service" for index in range(50))
    assert len(select_services(f"Compare {services}", CATALOG)) == MAX_SERVICES_PER_QUERY


def test_new_catalog_service_requires_no_source_change():
    assert select_services("inventory service response time", CATALOG) == ["InventorySvc"]
