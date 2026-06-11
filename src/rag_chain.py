"""RAG retrieval + generation chain.

Combines document retrieval and LLM generation using Groq.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from src.config import GROQ_MODEL, GROQ_API_KEY, TEMPERATURE, MAX_TOKENS

# RAG prompt template
RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful AI assistant. Answer the user's question based on the "
        "provided context. If the context doesn't contain enough information, "
        "say so and use your general knowledge to help.\n\n"
        "Context:\n{context}",
    ),
    ("human", "{question}"),
])


def create_llm() -> ChatGroq:
    """Create a Groq LLM instance."""
    return ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
