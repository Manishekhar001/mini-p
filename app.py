"""Streamlit frontend for the RAG Chatbot.

Features:
- Upload PDF/TXT documents for RAG indexing (no re-indexing on thread switch)
- Sidebar with thread management (new, select, delete)
- Auto-title threads from the first message
- RAG-powered chat with streaming
- Expandable citations showing retrieved context
- Resource overview showing indexed files
"""

import streamlit as st

from langchain_core.messages import HumanMessage

import traceback

from langgraph_backend import (
    chatbot,
    vector_store_manager,
    generate_thread_id,
    add_thread,
    update_thread_title,
    delete_thread,
    get_all_threads,
    load_conversation,
    clean_title,
)
from src.config import UPLOAD_DIR
from src.resource_manager import load_resources


# ---------- Page Config ----------

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)

# ---------- Setup Check ----------

# LLM: Groq (fast cloud inference, free tier).
# Embeddings: Ollama locally (nomic-embed-text).
#
# Make sure Ollama is running (`ollama serve`) with the embedding model:
#   ollama pull nomic-embed-text
#
# For the LLM, set your Groq API key in `.env`:
#   GROQ_API_KEY=gsk_...

# ---------- Session Initialization ----------

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()
    add_thread(st.session_state["thread_id"])

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "has_indexed" not in st.session_state:
    st.session_state["has_indexed"] = vector_store_manager.has_index()

if "indexed_files" not in st.session_state:
    if st.session_state["has_indexed"] and UPLOAD_DIR.exists():
        existing_files = {f.name for f in UPLOAD_DIR.iterdir() if f.is_file() and f.suffix.lower() in {".pdf", ".txt"}}
        st.session_state["indexed_files"] = existing_files
    else:
        st.session_state["indexed_files"] = set()


# ---------- Utility Functions ----------


def reset_chat() -> None:
    """Start a new chat thread."""
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []
    add_thread(thread_id)
    st.rerun()


def switch_thread(thread_id: str) -> None:
    """Switch to an existing chat thread and load its full state."""
    st.session_state["thread_id"] = thread_id

    config = {"configurable": {"thread_id": thread_id}}
    temp_messages = []

    try:
        state = chatbot.get_state(config)
        if state and state.values:
            messages = state.values.get("messages", [])
            last_context = state.values.get("retrieved_context", "")

            for msg in messages:
                role = "user" if msg.type == "human" else "assistant"
                temp_messages.append({"role": role, "content": msg.content})

            used_rag = state.values.get("use_rag", False)
            if used_rag and last_context and temp_messages and temp_messages[-1]["role"] == "assistant":
                temp_messages[-1]["citations"] = last_context
    except Exception as e:
        print(f"[switch_thread] get_state failed: {e}")
        traceback.print_exc()

    if not temp_messages:
        temp_messages = load_conversation(thread_id)
        temp_messages = [
            {"role": "user" if msg.type == "human" else "assistant", "content": msg.content}
            for msg in temp_messages
        ]

    st.session_state["message_history"] = temp_messages
    st.rerun()


# ---------- Auto-load Messages on Refresh ----------

if not st.session_state["message_history"]:
    config = {"configurable": {"thread_id": st.session_state["thread_id"]}}
    loaded = []
    try:
        state = chatbot.get_state(config)
        if state and state.values:
            messages = state.values.get("messages", [])
            last_context = state.values.get("retrieved_context", "")
            used_rag = state.values.get("use_rag", False)

            for msg in messages:
                role = "user" if msg.type == "human" else "assistant"
                loaded.append({"role": role, "content": msg.content})

            if used_rag and last_context and loaded and loaded[-1]["role"] == "assistant":
                loaded[-1]["citations"] = last_context

            st.session_state["message_history"] = loaded
    except Exception as e:
        print(f"[auto-load] get_state failed: {e}")
        traceback.print_exc()


# ---------- Sidebar ----------

with st.sidebar:
    st.title("🤖 RAG Chatbot")

    if st.button("➕ New Chat", use_container_width=True):
        reset_chat()

    st.divider()

    # Document Upload Section
    st.subheader("📄 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="file_uploader",
    )

    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state["indexed_files"]]

        if new_files:
            total_chunks = 0
            with st.spinner("Indexing documents..."):
                for uploaded_file in new_files:
                    file_path = UPLOAD_DIR / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    chunks = vector_store_manager.index_file(file_path)
                    if chunks > 0:
                        total_chunks += chunks
                        st.session_state["indexed_files"].add(uploaded_file.name)
                        st.toast(f"✓ {uploaded_file.name} ({chunks} chunks)", icon="✅")
                    else:
                        st.warning(f"⚠ {uploaded_file.name} — no content extracted")

            if total_chunks > 0:
                st.session_state["has_indexed"] = True
                st.toast(f"✅ Total: {total_chunks} chunks indexed.", icon="✅")
                st.rerun()
        else:
            st.info("All files already indexed. Upload new files to add more.")

    st.divider()

    # ---------- Resource Overview ----------
    st.subheader("📂 Indexed Resources")

    resources = load_resources()
    if resources:
        for filename, info in resources.items():
            desc = info.get("description", "")
            chunks = info.get("chunks", 0)
            label = f"**{filename}**"
            if desc:
                label += f"  \n{desc}"
            label += f"  \n`{chunks} chunks`"
            st.markdown(label)

        # Clear all button with confirmation
        with st.popover("🗑 Clear all", help="Remove all indexed files and start fresh"):
            st.warning("This will permanently delete all indexed files and their data.")
            if st.button("Yes, clear everything", type="primary", use_container_width=True):
                with st.spinner("Clearing all resources..."):
                    vector_store_manager.clear_all()
                    st.session_state["has_indexed"] = False
                    st.session_state["indexed_files"] = set()
                    st.toast("All resources cleared.", icon="✅")
                    st.rerun()
    elif st.session_state["has_indexed"]:
        st.caption("Resources exist but metadata unavailable. Re-index files.")
    else:
        st.caption("No files indexed yet.")

    st.divider()

    # Chat History
    st.subheader("💬 Chat History")
    threads = get_all_threads()

    if not threads:
        st.caption("No conversations yet. Start a new chat!")

    for thread in threads:
        col1, col2 = st.columns([4, 1])

        is_active = thread["thread_id"] == st.session_state["thread_id"]
        btn_label = f"📌 {thread['title']}" if is_active else thread["title"]

        if col1.button(
            btn_label,
            key=f"select_{thread['thread_id']}",
            use_container_width=True,
            disabled=is_active,
        ):
            switch_thread(thread["thread_id"])

        if col2.button(
            "🗑",
            key=f"delete_{thread['thread_id']}",
            help=f"Delete '{thread['title']}'",
        ):
            delete_thread(thread["thread_id"])
            st.rerun()




# ---------- Main Chat UI ----------

st.title("💬 RAG Chat")

if st.session_state["has_indexed"]:
    st.caption("📚 RAG ready — LLM judge will decide when to use your documents")
else:
    st.caption("💬 Chat mode — upload documents in the sidebar to enable RAG")

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("citations"):
            with st.expander("📄 Show sources"):
                citations_text = message["citations"]
                parts = citations_text.split("\n\n[Document")
                for i, part in enumerate(parts):
                    if i == 0 and not part.startswith("[Document"):
                        continue
                    if i > 0:
                        part = f"[Document{part}"
                    st.markdown(part)
                    st.divider()

user_input = st.chat_input("Type your message here...")

if user_input:
    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input,
    })
    with st.chat_message("user"):
        st.markdown(user_input)

    if len(st.session_state["message_history"]) == 1:
        title = clean_title(user_input)
        update_thread_title(st.session_state["thread_id"], title)

    config = {"configurable": {"thread_id": st.session_state["thread_id"]}}

    with st.chat_message("assistant"):
        placeholder = st.empty()
        status_placeholder = st.empty()
        status_placeholder.info("🤔 Thinking...")
        full_response = ""
        citations = ""
        used_rag = False
        first_token = True

        # Use stream_mode="values" so we get the full state after every
        # superstep. This lets us capture use_rag + retrieved_context
        # directly from the stream, avoiding a separate get_state() call,
        # AND show step-by-step status (searching → judging → generating).
        for event in chatbot.stream(
            {"messages": [HumanMessage(user_input)]},
            config=config,
            stream_mode="values",
        ):
            # Capture RAG fields as soon as they appear in the state
            used_rag = event.get("use_rag", used_rag)
            citations = event.get("retrieved_context", citations)

            # Show step-by-step status based on which fields appeared
            messages = event.get("messages", [])
            if messages and messages[-1].type == "ai":
                current_text = messages[-1].content or ""
                if current_text:
                    if first_token:
                        status_placeholder.empty()
                        first_token = False
                    full_response = current_text
                    placeholder.markdown(full_response + "▌")
            elif "use_rag" in event:
                status_placeholder.info("⚖️ Evaluating relevance...")
            elif "retrieved_context" in event:
                status_placeholder.info("🔍 Searching documents...")
            # else: still in initial "🤔 Thinking..." phase

        if first_token:
            status_placeholder.empty()
        placeholder.markdown(full_response)

    # Show RAG / plain chat indicator
    if used_rag:
        st.caption("📚 *Answered from your documents*")
    else:
        st.caption("💬 *Answered from general knowledge*")

    ai_entry = {
        "role": "assistant",
        "content": full_response,
    }
    if used_rag and citations:
        ai_entry["citations"] = citations

    st.session_state["message_history"].append(ai_entry)
