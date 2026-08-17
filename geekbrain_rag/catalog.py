from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

SERVICE_NAME_RE = re.compile(
    r"\b([A-Za-z][A-Za-z0-9_-]{1,62}(?:Svc|Service|GW|Gateway|Detector)(?:[_-]?[vV]?\d+)?)\b",
    re.IGNORECASE,
)


def normalized_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.replace("Đ", "D").replace("đ", "d"))
    ascii_text = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", ascii_text.lower()).strip()


def _service_terms(service: str) -> set[str]:
    """Generate human spellings from a catalog value without a domain alias table."""
    words = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", service).replace("-", " ").replace("_", " ")
    normalized = normalized_text(words)
    terms = {normalized, normalized.replace(" ", "")}
    suffixes = {
        " svc": [" service", " services", " svcs"],
        " gw": [" gateway", " gateways", " gws"],
        " service": [" services", " svc", " svcs"],
        " gateway": [" gateways", " gw", " gws"],
        " detector": [" detectors"],
    }
    for suffix, expansions in suffixes.items():
        if normalized.endswith(suffix):
            stem = normalized[: -len(suffix)]
            for expanded in expansions:
                terms.add(stem + expanded)
            terms.add(stem + suffix)
    return {term.strip() for term in terms if term.strip()}


def match_services(text: str, catalog: Iterable[str] = ()) -> list[str]:
    """Resolve services from a discovered catalog, while retaining explicit new names."""
    normalized = f" {normalized_text(text)} "
    found: list[str] = []
    for service in catalog:
        if any(f" {term} " in normalized for term in _service_terms(service)):
            found.append(service)
    canonical_by_lower = {service.lower(): service for service in catalog}
    for match in SERVICE_NAME_RE.finditer(text):
        candidate = match.group(1)
        if candidate.lower() in {"microservice", "service", "gateway", "services", "gateways"}:
            continue
        found.append(canonical_by_lower.get(candidate.lower(), candidate))
    return list(dict.fromkeys(found))
