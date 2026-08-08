"""Safe compatibility tools with no arbitrary SQL execution surface."""

from langchain_core.tools import tool

from geekbrain_rag.analytics import AnalyticsEngine
from geekbrain_rag.config import get_settings
from geekbrain_rag.monitoring import MonitoringClient
from geekbrain_rag.retrieval import KnowledgeBaseRetriever

settings = get_settings()


@tool
def analytics_query(question: str) -> str:
    """Answer an analytics question using the read-only, allowlisted GeekBrain database."""
    return AnalyticsEngine(settings).query(question).content


@tool
def live_metrics(question: str) -> str:
    """Get live metrics for explicit services, or all services for rankings/comparisons."""
    return MonitoringClient(settings).query(question).content


@tool
def search_documents(query: str) -> str:
    """Search governed CURRENT documents in the Bedrock Knowledge Base."""
    docs = KnowledgeBaseRetriever(settings).retrieve(query)
    return "\n\n".join(f"Source: {doc.source}\n{doc.content}" for doc in docs)


tools = [analytics_query, live_metrics, search_documents]
