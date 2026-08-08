import uuid

from langchain_core.messages import HumanMessage

from graph import app


def ask(question: str, session_id: str | None = None):
    """Hỏi câu hỏi và trả lời"""
    if session_id is None:
        session_id = str(uuid.uuid4())  # Mỗi lần chạy mới là conversation mới

    config = {"configurable": {"thread_id": session_id}}

    print(f"\n👤 User: {question}")
    result = app.invoke({"messages": [HumanMessage(content=question)]}, config=config)

    answer = result["messages"][-1].content
    print(f"🤖 AI: {answer}")
    return answer


# ================== TEST CÁC LEVEL ==================
if __name__ == "__main__":
    print("🚀 GeekBrain AI Agent đang chạy...\n")

    # Test L1
    ask("Who is the Team Platform lead?")

    # Test L2 (conflict)
    ask("What is the API rate limit for PaymentGW?")

    # Test L3 - Historical
    ask("What was PaymentGW's total infrastructure cost in Q1 2026?")

    # Test L3 - Current
    ask("What is PaymentGW's current p99 latency?")

    # Test L4 - Memory (multi-turn)
    session = "conversation-test-001"
    ask("Which service had the highest infrastructure cost in March 2026?", session)
    ask("What was the main cause of that service's cost increase?", session)
    ask("Which team is responsible for that service?", session)

    # Test Bonus B - Agent Reasoning
    print("\n================== TEST BONUS B ==================")
    bonus_session = "bonus-test-001"
    ask(
        "Is PaymentGW in a healthy state? Assess its reliability and flag anything that needs attention. Include data from both metrics and SLAs.",
        bonus_session,
    )

    print("\n✅ Tất cả test đã chạy xong!")
