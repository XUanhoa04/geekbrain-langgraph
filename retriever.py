"""Compatibility module. New code should use geekbrain_rag.retrieval."""

from langchain_aws import AmazonKnowledgeBasesRetriever, ChatBedrockConverse

from geekbrain_rag.config import get_settings

settings = get_settings()
llm = ChatBedrockConverse(
    model_id=settings.model_id,
    region_name=settings.aws_region,
    temperature=0,
    max_tokens=2400,
)
retriever = (
    AmazonKnowledgeBasesRetriever(
        knowledge_base_id=settings.bedrock_kb_id,
        retrieval_config={
            "vectorSearchConfiguration": {
                "numberOfResults": settings.rag_retrieval_top_k,
                "overrideSearchType": "SEMANTIC",
            }
        },
    )
    if settings.bedrock_kb_id
    else None
)
