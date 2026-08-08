from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import boto3
from langchain_core.messages import HumanMessage

from geekbrain_rag.agent import build_graph
from geekbrain_rag.config import Settings, get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Bedrock grounding scores for one RAG answer")
    parser.add_argument("question")
    args = parser.parse_args()
    settings = get_settings()
    unguarded = Settings(bedrock_guardrail_id="")
    graph = build_graph(unguarded)
    result = graph.invoke(
        {"messages": [HumanMessage(content=args.question)]},
        {"configurable": {"thread_id": f"guardrail-diagnostic-{uuid.uuid4()}"}},
    )
    answer = str(result["messages"][-1].content)
    context = result.get("retrieved_context") or ""
    client = boto3.client("bedrock-runtime", region_name=settings.aws_region)
    response = client.apply_guardrail(
        guardrailIdentifier=settings.bedrock_guardrail_id,
        guardrailVersion=settings.bedrock_guardrail_version,
        source="OUTPUT",
        content=[
            {"text": {"text": context[:80_000], "qualifiers": ["grounding_source"]}},
            {"text": {"text": args.question, "qualifiers": ["query"]}},
            {"text": {"text": answer, "qualifiers": ["guard_content"]}},
        ],
        outputScope="FULL",
    )
    assessments = [item.get("contextualGroundingPolicy") for item in response.get("assessments", [])]
    print(answer)
    print(json.dumps({"action": response.get("action"), "grounding": assessments}, indent=2))


if __name__ == "__main__":
    main()
