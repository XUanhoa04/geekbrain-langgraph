from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import boto3

from .config import Settings

CREDENTIAL_SECRET_RE = re.compile(
    r"\b(aws access key(?: id)?|secret access key|secret key|password|passwords|credential|"
    r"credentials|api key|access token|refresh token|mat khau|khoa api|khoa truy cap|"
    r"thong tin dang nhap|ma thong bao|token)\b"
)
STRONG_EXFILTRATION_RE = re.compile(
    r"\b(show|reveal|print|dump|expose|send|share|provide|give me|hand me|can i have|"
    r"display|leak|hien thi|tiet lo|in ra|xuat ra|gui cho toi|dua cho toi|dua toi|"
    r"cho toi|doc ra|lam lo)\b"
)
SECRET_VALUE_QUESTION_RE = re.compile(
    r"\b(what is|tell me|give me|cho toi|lay cho toi)\b.{0,60}\b(actual|current|raw|value|"
    r"content|la gi|gia tri|noi dung)?\b"
)


def _security_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.replace("Đ", "D").replace("đ", "d"))
    ascii_text = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", ascii_text.lower()).strip()


def requests_credential_exfiltration(text: str) -> bool:
    normalized = _security_text(text)
    secret_matches = list(CREDENTIAL_SECRET_RE.finditer(normalized))
    if not secret_matches:
        return False
    for secret in secret_matches:
        window = normalized[max(0, secret.start() - 100) : secret.end() + 100]
        operational_guidance = bool(
            re.search(
                r"\b(rotate|rotation|protect|store|manage|policy|quy trinh|bao ve|quan ly|"
                r"thay doi|luu tru|xoay vong|example|format|vi du|dinh dang)\b",
                window,
            )
        )
        conceptual_question = bool(
            re.fullmatch(
                r"(?:what is|define) (?:an? )?(?:api key|password|credential|access token|"
                r"refresh token)|(?:api key|mat khau|thong tin dang nhap|token) la gi",
                normalized,
            )
        )
        strong_actions = {match.group(0) for match in STRONG_EXFILTRATION_RE.finditer(window)}
        if strong_actions:
            harmless_display = strong_actions.issubset(
                {"show", "display", "hien thi", "provide", "cho toi"}
            )
            explicit_value = bool(
                re.search(r"\b(actual|current|raw|value|content|gia tri|noi dung)\b", window)
            )
            if not (operational_guidance and harmless_display and not explicit_value):
                return True
        if (
            SECRET_VALUE_QUESTION_RE.search(window)
            and not operational_guidance
            and not conceptual_question
        ):
            return True
        if not operational_guidance and re.search(
            r"\b(mat khau|api key|secret key|token)\b.{0,30}\b(la gi|o dau)\b", window
        ) and not conceptual_question:
            return True
    return False


@dataclass(slots=True)
class GuardrailResult:
    blocked: bool
    text: str
    action: str = "NONE"


class Guardrails:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = boto3.client("bedrock-runtime", region_name=settings.aws_region)

    @property
    def enabled(self) -> bool:
        return bool(self.settings.bedrock_guardrail_id)

    def check_input(self, text: str) -> GuardrailResult:
        if requests_credential_exfiltration(text):
            return GuardrailResult(
                True,
                "Yêu cầu tiết lộ thông tin xác thực hoặc bí mật đã bị chặn.",
                "LOCAL_CREDENTIAL_POLICY",
            )
        if not self.enabled:
            return GuardrailResult(False, text)
        response = self.client.apply_guardrail(
            guardrailIdentifier=self.settings.bedrock_guardrail_id,
            guardrailVersion=self.settings.bedrock_guardrail_version,
            source="INPUT",
            content=[{"text": {"text": text}}],
            outputScope="INTERVENTIONS",
        )
        action = response.get("action", "NONE")
        output = " ".join(item.get("text", "") for item in response.get("outputs", [])) or text
        return GuardrailResult(action == "GUARDRAIL_INTERVENED", output, action)

    def check_grounding(self, query: str, grounding_source: str, answer: str) -> GuardrailResult:
        if not self.enabled or not grounding_source.strip():
            return GuardrailResult(False, answer)
        response = self.client.apply_guardrail(
            guardrailIdentifier=self.settings.bedrock_guardrail_id,
            guardrailVersion=self.settings.bedrock_guardrail_version,
            source="OUTPUT",
            content=[
                {"text": {"text": grounding_source[:80_000], "qualifiers": ["grounding_source"]}},
                {"text": {"text": query[:10_000], "qualifiers": ["query"]}},
                {"text": {"text": answer[:20_000], "qualifiers": ["guard_content"]}},
            ],
            outputScope="FULL",
        )
        action = response.get("action", "NONE")
        output = " ".join(item.get("text", "") for item in response.get("outputs", [])) or answer
        assessments = response.get("assessments", [])
        contextual_filters = [
            item
            for assessment in assessments
            for item in assessment.get("contextualGroundingPolicy", {}).get("filters", [])
        ]
        grounding_detected = any(
            item.get("type") == "GROUNDING" and item.get("detected")
            for item in contextual_filters
        )
        relevance_detected = any(
            item.get("type") == "RELEVANCE" and item.get("detected")
            for item in contextual_filters
        )

        def has_detected_policy(value: object) -> bool:
            if isinstance(value, dict):
                return value.get("detected") is True or any(
                    has_detected_policy(item) for item in value.values()
                )
            if isinstance(value, list):
                return any(has_detected_policy(item) for item in value)
            return False

        other_policy_detected = any(
            has_detected_policy(
                {key: value for key, value in assessment.items() if key != "contextualGroundingPolicy"}
            )
            for assessment in assessments
        )
        # Bedrock's relevance filter is calibrated for short answers and can flag a
        # grounded subsection of a broad, multi-part report. Relevance-only findings
        # trigger no destructive rewrite; unsupported grounding and all other safety
        # policies continue to block.
        blocked = bool(
            grounding_detected
            or other_policy_detected
            or (action == "GUARDRAIL_INTERVENED" and not relevance_detected)
        )
        return GuardrailResult(blocked, output if blocked else answer, action)

    def check_claim_support(
        self, query: str, grounding_source: str, claim: str
    ) -> GuardrailResult:
        """At claim granularity, enforce support without misusing whole-answer relevance."""
        if not self.enabled or not grounding_source.strip():
            return GuardrailResult(False, claim)
        response = self.client.apply_guardrail(
            guardrailIdentifier=self.settings.bedrock_guardrail_id,
            guardrailVersion=self.settings.bedrock_guardrail_version,
            source="OUTPUT",
            content=[
                {"text": {"text": grounding_source[:80_000], "qualifiers": ["grounding_source"]}},
                {"text": {"text": query[:10_000], "qualifiers": ["query"]}},
                {"text": {"text": claim[:20_000], "qualifiers": ["guard_content"]}},
            ],
            outputScope="FULL",
        )
        grounding_filters = [
            item
            for assessment in response.get("assessments", [])
            for item in assessment.get("contextualGroundingPolicy", {}).get("filters", [])
            if item.get("type") == "GROUNDING"
        ]
        unsupported = any(item.get("detected") for item in grounding_filters)
        return GuardrailResult(unsupported, claim, "CLAIM_GROUNDING")
