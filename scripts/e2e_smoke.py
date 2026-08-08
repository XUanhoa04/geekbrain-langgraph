from __future__ import annotations

import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.messages import HumanMessage

from graph import app

QUESTIONS = [
    "Is PaymentGW's current error rate within its SLA target?",
    "Which service had the highest total cost in March 2026?",
    "Which service currently handles the most requests per minute?",
]


def main() -> None:
    for question in QUESTIONS:
        response = app.invoke(
            {"messages": [HumanMessage(content=question)]},
            {"configurable": {"thread_id": f"e2e-{uuid.uuid4()}"}},
        )
        print(f"QUESTION: {question}\nANSWER: {response['messages'][-1].content}\n")


if __name__ == "__main__":
    main()
