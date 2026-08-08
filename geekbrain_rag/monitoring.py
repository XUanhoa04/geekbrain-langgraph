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
MAX_SERVICES_PER_QUERY = 20
SERVICE_NAME_RE = re.compile(
    r"\b([A-Za-z][A-Za-z0-9-]{1,48}(?:Svc|Service|GW|Gateway|Detector))\b",
    re.IGNORECASE,
)


def select_services(question: str) -> list[str]:
    explicit = [service for service in SERVICES if service.lower() in question.lower()]
    lowered = question.lower()
    aliases = {
        "payment gateway": "PaymentGW",
        "notification service": "NotificationSvc",
        "order service": "OrderSvc",
        "reporting service": "ReportingSvc",
        "auth service": "AuthSvc",
    }
    explicit.extend(canonical for alias, canonical in aliases.items() if alias in lowered)
    explicit.extend(match.group(1) for match in SERVICE_NAME_RE.finditer(question))
    explicit = list(dict.fromkeys(explicit))
    comparative = bool(
        re.search(
            r"(?i)\b(all|across|compare|comparison|highest|lowest|most|least|rank|every|"
            r"tat ca|so sanh|cao nhat|thap nhat|xep hang)\b",
            question,
        )
    )
    if comparative:
        return list(dict.fromkeys([*SERVICES, *explicit]))[:MAX_SERVICES_PER_QUERY]
    if not explicit:
        return list(SERVICES)
    return explicit[:MAX_SERVICES_PER_QUERY]


@dataclass(slots=True)
class MonitoringClient:
    settings: Settings

    def query(self, question: str) -> Evidence:
        services = select_services(question)
        include_status = bool(
            re.search(
                r"(?i)\b(healthy|health|status|running normally|degraded|active alerts?|reliable|"
                r"reliability|report card|current data|trạng thái|sức khỏe|ổn định|cảnh báo|"
                r"trang thai|suc khoe|on dinh|canh bao)\b",
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
                    if not isinstance(metrics, dict):
                        raise TypeError("Monitoring metrics response must be a JSON object")
                    if include_status:
                        status_response = session.get(
                            f"{self.settings.monitoring_api_url.rstrip('/')}/status/{quote(service)}",
                            timeout=(2, 8),
                        )
                        status_response.raise_for_status()
                        status_payload = status_response.json()
                        if not isinstance(status_payload, dict):
                            raise TypeError("Monitoring status response must be a JSON object")
                        metrics["service_status"] = status_payload
                    result[service] = metrics
                except (requests.RequestException, TypeError, ValueError) as exc:
                    errors[service] = str(exc)
        payload = {"observed_services": result, "errors": errors}
        kind = "LIVE_METRICS" if result else "LIVE_METRICS_ERROR"
        return Evidence(kind, json.dumps(payload, ensure_ascii=False), "Monitoring API")
