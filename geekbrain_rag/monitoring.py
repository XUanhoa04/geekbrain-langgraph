from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from urllib.parse import quote

import requests

from .catalog import match_services
from .config import Settings
from .retrieval import Evidence

MAX_SERVICES_PER_QUERY = 20


def select_services(question: str, catalog: tuple[str, ...] = ()) -> list[str]:
    explicit = match_services(question, catalog)
    comparative = bool(
        re.search(
            r"(?i)\b(all|across|compare|comparison|highest|lowest|most|least|rank|every|"
            r"tat ca|so sanh|cao nhat|thap nhat|xep hang)\b",
            question,
        )
    )
    if comparative:
        return list(dict.fromkeys([*catalog, *explicit]))[:MAX_SERVICES_PER_QUERY]
    if not explicit:
        return list(catalog)[:MAX_SERVICES_PER_QUERY]
    return explicit[:MAX_SERVICES_PER_QUERY]


@dataclass(slots=True)
class MonitoringClient:
    settings: Settings
    _service_cache: tuple[str, ...] | None = field(default=None, init=False, repr=False)

    def available_services(self) -> tuple[str, ...]:
        """Discover the monitoring inventory from the API instead of source code."""
        if self._service_cache is not None:
            return self._service_cache
        try:
            response = requests.get(
                f"{self.settings.monitoring_api_url.rstrip('/')}/services",
                timeout=(2, 5),
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                raise TypeError("Monitoring service catalog must be a JSON list")
            services = tuple(
                dict.fromkeys(
                    str(value)
                    for value in payload
                    if isinstance(value, str)
                    and re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{1,63}", value)
                )
            )
            self._service_cache = services[:MAX_SERVICES_PER_QUERY]
        except (requests.RequestException, TypeError, ValueError):
            self._service_cache = ()
        return self._service_cache

    def query(self, question: str) -> Evidence:
        catalog = self.available_services()
        services = select_services(question, catalog)
        if not services:
            return Evidence(
                "LIVE_METRICS_ERROR",
                json.dumps(
                    {
                        "observed_services": {},
                        "errors": {"catalog": "service discovery unavailable"},
                    }
                ),
                "Monitoring API",
                metadata={"reason": "SERVICE_DISCOVERY_UNAVAILABLE"},
            )
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
