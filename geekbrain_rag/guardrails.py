from __future__ import annotations

import re
from dataclasses import dataclass

import boto3

from .config import Settings

CREDENTIAL_EXFILTRATION_RE = re.compile(
    r"(?i)\b(show|reveal|give|print|dump|expose|send|tell\s+me)\b.{0,80}"
    r"\b(aws\s+access\s+keys?|secret\s+(?:access\s+)?keys?|passwords?|credentials?|api\s+keys?)\b"
)


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
        if CREDENTIAL_EXFILTRATION_RE.search(text):
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
            outputScope="INTERVENTIONS",
        )
        action = response.get("action", "NONE")
        output = " ".join(item.get("text", "") for item in response.get("outputs", [])) or answer
        return GuardrailResult(action == "GUARDRAIL_INTERVENED", output, action)
