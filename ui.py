import html
import re
import uuid
from collections import Counter

import streamlit as st
from langchain_core.messages import HumanMessage

from graph import app


def get_relevant_snippet(chunk_text, answer):
    # Dùng regex để loại bỏ các ký tự không phải chữ/số, chuyển thành chữ thường
    clean_ans = re.sub(r"[^\w\s]", "", answer).lower()
    stopwords = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "this",
        "that",
        "it",
        "they",
        "we",
        "you",
        "i",
        "he",
        "she",
        "as",
        "be",
        "have",
        "has",
        "had",
    }
    # Lấy các từ khóa thực sự
    keywords = {w for w in clean_ans.split() if w not in stopwords and len(w) > 1}

    if not keywords:
        return chunk_text[:200] + ("..." if len(chunk_text) > 200 else "")

    chunk_clean = chunk_text.lower()

    # Tính tần suất (term frequency) trong chunk
    words_in_chunk = re.findall(r"\b\w+\b", chunk_clean)
    word_freq = Counter(words_in_chunk)

    lines = chunk_text.split("\n")
    best_line_idx = -1
    best_score = -1

    for i, line in enumerate(lines):
        line_clean = line.lower()
        score = 0
        for k in keywords:
            if re.search(r"\b" + re.escape(k) + r"\b", line_clean):
                freq = word_freq.get(k, 1)
                # Điểm tỉ lệ nghịch với tần suất (tương tự IDF)
                score += 100.0 / freq

        if score > best_score and line.strip() != "":
            best_score = score
            best_line_idx = i

    if best_line_idx == -1:
        snippet = chunk_text[:200]
    else:
        # Lấy 1 dòng trước, dòng hiện tại, và 2 dòng sau
        start_idx = max(0, best_line_idx - 1)
        end_idx = min(len(lines), best_line_idx + 3)
        snippet = "\n".join(lines[start_idx:end_idx])

    # Làm đẹp kết quả
    snippet = snippet.strip()
    if not chunk_text.startswith(snippet):
        snippet = "... " + snippet
    if not chunk_text.endswith(snippet):
        snippet = snippet + " ..."

    return snippet


def remove_private_reasoning(text):
    """Drop model-only thinking blocks before rendering or storing an answer."""
    return re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL).strip()


def source_items(raw_sources, answer):
    """Return the cited, de-duplicated source excerpts used by an answer."""
    items = []
    seen = set()

    for raw_source in raw_sources:
        for chunk_index, raw_chunk in enumerate(raw_source.split("\n\n---\n\n"), start=1):
            chunk = raw_chunk.strip()
            if not chunk:
                continue

            lines = chunk.splitlines()
            heading = lines[0].strip()
            numbered = re.match(r"^\[(\d+)\]\s+Source:\s*(.+)$", heading)
            named = re.match(r"^\[Source:\s*(.+)\]$", heading)
            evidence = re.match(
                r"^\[(\d+)\]\s+kind=([^;]+);\s+source=([^;]+);",
                heading,
            )

            if evidence:
                citation, _kind, source_name = evidence.groups()
                if f"[{citation}]" not in answer and source_name not in answer:
                    continue
                title = f"[{citation}] {source_name.strip()}"
                content = "\n".join(lines[1:]).strip()
            elif numbered:
                citation, source_name = numbered.groups()
                if f"[{citation}]" not in answer and source_name not in answer:
                    continue
                title = f"[{citation}] {source_name.strip()}"
                content = "\n".join(lines[1:]).strip()
            elif named:
                source_name = named.group(1).strip()
                if source_name not in answer:
                    continue
                title = source_name
                content = "\n".join(lines[1:]).strip()
            else:
                title = f"Retrieved evidence {chunk_index}"
                content = chunk

            key = (title, content)
            if content and key not in seen:
                seen.add(key)
                items.append((title, get_relevant_snippet(content, answer)))

    return items


def render_sources(raw_sources, answer):
    items = source_items(raw_sources, answer)
    if not items:
        return

    with st.popover(f"Sources · {len(items)}"):
        st.caption("Only evidence referenced by this answer is shown.")
        for title, excerpt in items:
            with st.expander(title):
                st.markdown(excerpt)


# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="AegisRAG Workspace",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ================== CUSTOM CSS ==================
st.markdown(
    """
<style>
    :root {
        --canvas: #f5f5f0;
        --surface: #ffffff;
        --surface-muted: #f0f1eb;
        --ink: #20231f;
        --muted: #687068;
        --line: #dfe2da;
        --accent: #276247;
        --accent-soft: #e3eee7;
        --amber: #a96d22;
    }

    html, body, [class*="css"] {
        font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .stApp {
        background: var(--canvas);
        color: var(--ink);
    }

    .block-container {
        max-width: 920px;
        padding-top: 2.5rem;
        padding-bottom: 7rem;
    }

    section[data-testid="stSidebar"] {
        background: #eceee8 !important;
        border-right: 1px solid var(--line) !important;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    .main-header {
        padding: 0.5rem 0 1.5rem;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid var(--line);
    }

    .main-header h1 {
        margin: 0;
        color: var(--ink);
        font-size: clamp(1.8rem, 4vw, 2.35rem);
        font-weight: 680;
        letter-spacing: -0.04em;
    }

    .main-header p {
        margin: 0.45rem 0 0;
        color: var(--muted);
        font-size: 1rem;
    }

    .stChatMessage {
        background: var(--surface) !important;
        border: 1px solid var(--line);
        border-radius: 10px !important;
        margin-bottom: 0.8rem !important;
        box-shadow: 0 1px 2px rgba(32, 35, 31, 0.04);
    }

    .trace-card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.85rem;
        margin-bottom: 0.75rem;
    }

    .activity-meta {
        color: var(--muted);
        font-size: 0.76rem;
    }

    .metric-card {
        background: transparent;
        border-top: 1px solid var(--line);
        padding: 0.8rem 0;
        margin: 0.25rem 0;
    }

    .metric-value {
        color: var(--ink);
        font-size: 1.35rem;
        font-weight: 650;
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.76rem;
        margin-top: 0.1rem;
    }

    .stChatInputContainer {
        background: var(--surface) !important;
        border: 1px solid var(--line) !important;
        border-radius: 10px !important;
        box-shadow: 0 6px 24px rgba(32, 35, 31, 0.08) !important;
    }

    .stChatInputContainer:focus-within {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px var(--accent-soft) !important;
    }

    .stButton > button {
        border-color: var(--line);
        border-radius: 8px;
        background: var(--surface);
        color: var(--ink);
    }

    .stButton > button:hover {
        border-color: var(--accent);
        color: var(--accent);
    }

    @media (max-width: 640px) {
        .block-container {
            padding-top: 1.25rem;
        }
        .main-header {
            padding-top: 0;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)

# ================== SESSION STATE ==================
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "activity" not in st.session_state:
    st.session_state.activity = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0

# ================== SIDEBAR ==================
with st.sidebar:
    st.markdown("### AegisRAG")
    st.caption("Governed knowledge workspace")

    if st.button("New conversation", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.activity = []
        st.session_state.total_queries = 0
        st.rerun()

    st.divider()
    st.markdown("#### Session")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""<div class="metric-card">
            <div class="metric-value">{st.session_state.total_queries}</div>
            <div class="metric-label">Queries</div>
        </div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""<div class="metric-card">
            <div class="metric-value">{len(st.session_state.messages) // 2}</div>
            <div class="metric-label">Turns</div>
        </div>""",
            unsafe_allow_html=True,
        )

    st.caption(f"Thread `{st.session_state.session_id[:8]}`")
    st.caption("Model `Claude Haiku 4.5`")
    st.divider()

    st.markdown("#### Recent questions")
    if st.session_state.activity:
        for item in reversed(st.session_state.activity[-4:]):
            safe_query = html.escape(item["query"])
            source_label = f'{item["sources"]} sources' if item["sources"] else "No sources"
            st.markdown(
                f'<div class="trace-card"><b>{safe_query}</b><br>'
                f'<span class="activity-meta">{source_label}</span></div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("Your latest questions will appear here.")

# ================== MAIN AREA ==================
st.markdown(
    """
<div class="main-header">
    <h1>Ask across your operations</h1>
    <p>Policies, services, costs and live metrics — grounded in governed sources.</p>
</div>
""",
    unsafe_allow_html=True,
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_sources(message.get("sources", []), message["content"])

if prompt := st.chat_input("Ask about a service, policy, cost or live metric"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.total_queries += 1

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        config = {"configurable": {"thread_id": st.session_state.session_id}}
        current_sources = []

        with st.spinner("Checking governed sources…"):
            for event in app.stream(
                {"messages": [HumanMessage(content=prompt)]},
                config=config,
                stream_mode="updates",
            ):
                for state_update in event.values():
                    messages = state_update.get("messages", [])
                    if isinstance(messages, list) and messages:
                        last_message = messages[-1]
                        if (
                            getattr(last_message, "name", None) == "search_documents"
                            and last_message.content
                        ):
                            current_sources.append(last_message.content)

                    if state_update.get("retrieved_context"):
                        current_sources.append(state_update["retrieved_context"])

            final_state = app.get_state(config)
            raw_answer = final_state.values["messages"][-1].content
            answer = remove_private_reasoning(raw_answer)

        st.markdown(answer)
        render_sources(current_sources, answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": current_sources}
    )
    st.session_state.activity.append(
        {
            "query": prompt[:72] + ("…" if len(prompt) > 72 else ""),
            "sources": len(source_items(current_sources, answer)),
        }
    )
    st.rerun()
