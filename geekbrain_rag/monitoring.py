from __future__ import annotations

import json
import re
from dataclasses import dataclass
from urllib.parse import quote

import requests

from .config import Settings
from .retrieval import Evidence

SERVICES = (
    "PaymentGW",
    "AuthSvc",
    "OrderSvc",
    "NotificationSvc",
    "FraudDetector",
    "ReportingSvc",
)


def select_services(question: str) -> list[str]:
    explicit = [service for service in SERVICES if service.lower() in question.lower()]
    comparative = bool(
        re.search(
            r"(?i)\b(all|across|compare|comparison|highest|lowest|most|least|rank|every)\b",
            question,
        )
    )
    if comparative or not explicit:
        return list(SERVICES)
    return explicit


@dataclass(slots=True)
class MonitoringClient:
    settings: Settings

    def query(self, question: str) -> Evidence:
        services = select_services(question)
        include_status = bool(
            re.search(
                r"(?i)\b(healthy|health|status|running normally|degraded|active alerts?|reliable|"
                r"reliability|report card|current data)\b",
                question,
            )
        )
        result: dict[str, object] = {}
        errors: dict[str, str] = {}
        with requests.Session() as session:
            for service in services:
                try:
                    response = session.get(
                        f"{self.settings.monitoring_api_url.rstrip('/')}/metrics/{quote(service)}",
                        timeout=(2, 8),
                    )
                    response.raise_for_status()
                    metrics = response.json()
                    if include_status:
                        status_response = session.get(
                            f"{self.settings.monitoring_api_url.rstrip('/')}/status/{quote(service)}",
                            timeout=(2, 8),
                        )
                        status_response.raise_for_status()
                        metrics["service_status"] = status_response.json()
                    result[service] = metrics
                except (requests.RequestException, ValueError) as exc:
                    errors[service] = str(exc)
        payload = {"observed_services": result, "errors": errors}
        kind = "LIVE_METRICS" if result else "LIVE_METRICS_ERROR"
        return Evidence(kind, json.dumps(payload, ensure_ascii=False), "Monitoring API")
