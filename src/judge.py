"""LLM Judge module.

Determines whether retrieved document context is relevant enough to
warrant a RAG response, or whether the query should be answered with
plain LLM knowledge instead.

Uses the resource metadata summary to understand what documents are
available, helping it make better routing decisions.
"""

from langchain_core.prompts import ChatPromptTemplate

from src.rag_chain import create_llm

# Judge prompt: evaluates if retrieved context is relevant to the question
JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a strict relevance judge. Given a user question, some "
        "retrieved document context, and a summary of available documents, "
        "determine if the context is useful for answering the question.\n\n"
        "Criteria for RELEVANT:\n"
        "- The context contains information that directly helps answer the question\n"
        "- The context is not empty or nonsensical\n"
        "- The question is about the content in the available documents\n"
        "- The question matches the topic of the available documents\n\n"
        "Criteria for NOT_RELEVANT:\n"
        "- The context is empty\n"
        "- The context is gibberish or irrelevant to the question\n"
        "- The question is a general greeting, casual chat, or unrelated topic\n"
        "- The question is clearly outside the scope of the available documents\n\n"
        "Available documents summary:\n{resource_summary}\n\n"
        "Reply with ONLY one word: 'RELEVANT' or 'NOT_RELEVANT'.\n"
        "Do not add any explanation or punctuation.",
    ),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
])


def judge_relevance(
    question: str,
    context: str,
    resource_summary: str = "",
) -> bool:
    """Judge whether the retrieved context is relevant to the question.

    Args:
        question: The user's question.
        context: Retrieved document context (may be empty).
        resource_summary: Summary of available documents for context.

    Returns:
        True if context is relevant, False if not.
    """
    if not context or not context.strip():
        return False

    llm = create_llm()
    chain = JUDGE_PROMPT | llm
    response = chain.invoke({
        "context": context,
        "question": question,
        "resource_summary": resource_summary or "No documents available.",
    })

    verdict = response.content.strip().upper()
    return "RELEVANT" in verdict
