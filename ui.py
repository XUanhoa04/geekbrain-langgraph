import re
import uuid

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
    from collections import Counter

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

    .trace-step {
        border-left: 2px solid var(--accent);
        padding: 0.35rem 0 0.35rem 0.65rem;
        margin: 0.45rem 0;
        color: var(--muted);
        font-size: 0.82rem;
    }

    .trace-step-rag { border-left-color: var(--accent); }
    .trace-step-tool { border-left-color: var(--amber); }
    .trace-step-router { border-left-color: #63758a; }

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
if "traces" not in st.session_state:
    st.session_state.traces = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0

# ================== SIDEBAR ==================
with st.sidebar:
    st.markdown("### AegisRAG")
    st.caption("Governed knowledge workspace")

    if st.button("New conversation", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.traces = []
        st.session_state.total_queries = 0
        st.rerun()

    st.divider()

    # Session info
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

    # Trace panel in sidebar
    st.markdown("#### Activity")
    if st.session_state.traces:
        for trace in st.session_state.traces:
            st.markdown(trace, unsafe_allow_html=True)

        import streamlit.components.v1 as components

        components.html(
            """
            <script>
                const sidebar = window.parent.document.querySelector('section[data-testid="stSidebar"]');
                if (sidebar) {
                    const scrollableDiv = sidebar.querySelector('div[data-testid="stSidebarUserContent"]') || sidebar.firstElementChild;
                    if (scrollableDiv) {
                        scrollableDiv.scrollTo({ top: scrollableDiv.scrollHeight, behavior: 'smooth' });
                    }
                }
            </script>
            """,
            height=0,
            width=0,
        )
    else:
        st.caption("Activity will appear after your first question.")

# ================== MAIN AREA ==================
# Header
st.markdown(
    """
<div class="main-header">
    <h1>Ask across your operations</h1>
    <p>Policies, services, costs and live metrics — grounded in governed sources.</p>
</div>
""",
    unsafe_allow_html=True,
)

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("reasoning"):
            with st.expander("🤔 **Final Reasoning** (Giải thích lựa chọn dữ liệu)"):
                st.info(msg["reasoning"])
        st.markdown(msg["content"])
        # Hiển thị nguồn tài liệu dạng popover nếu có
        if msg.get("sources"):
            with st.popover("📚 Nhấn để xem tài liệu trích xuất (Sources)"):
                for i, doc in enumerate(msg["sources"]):
                    chunks = doc.split("\n\n---\n\n")
                    valid_chunks = []
                    for idx, chunk in enumerate(chunks):
                        chunk = chunk.strip()
                        if not chunk:
                            continue

                        lines = chunk.split("\n")
                        import re

                        title_line = lines[0]
                        match = re.match(r"^\[(\d+)\] Source:\s*(.+)$", title_line)
                        if match:
                            chunk_idx = match.group(1)
                            filename = match.group(2).strip()
                            title = f"[{chunk_idx}] {filename}"
                            content = "\n".join(lines[1:])
                            full_text = msg["content"] + "\n" + msg.get("reasoning", "")
                            if f"[{chunk_idx}]" not in full_text and filename not in full_text:
                                continue
                        else:
                            title = (
                                title_line
                                if title_line.startswith("[Source:")
                                else f"Chunk {idx + 1}"
                            )
                            content = (
                                "\n".join(lines[1:]) if title_line.startswith("[Source:") else chunk
                            )
                            if title.startswith("[Source:"):
                                filename = title.replace("[Source:", "").replace("]", "").strip()
                                full_text = msg["content"] + "\n" + msg.get("reasoning", "")
                                if filename not in full_text:
                                    continue

                        snippet = get_relevant_snippet(content, msg["content"])
                        valid_chunks.append((title, snippet))

                    if valid_chunks:
                        st.markdown(f"**Tài liệu được trích xuất (Lần {i + 1}):**")
                        for title, content in valid_chunks:
                            with st.expander(f"📄 {title}"):
                                st.markdown(content)

# Chat input (at the bottom)
if prompt := st.chat_input("Ask AegisRAG about teams, costs, live metrics, or policies..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.total_queries += 1

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        config = {"configurable": {"thread_id": st.session_state.session_id}}

        # Clear previous traces for this query
        current_traces = []
        current_sources = []  # Lưu các text lấy từ RAG/Tool
        used_tools = set()

        with st.status("🔄 **Quá trình Reasoning...** (Live)", expanded=True) as status:
            # Stream events to capture trace
            for event in app.stream(
                {"messages": [HumanMessage(content=prompt)]}, config=config, stream_mode="updates"
            ):
                for node_name, state_update in event.items():
                    # Determine trace style
                    if node_name == "router":
                        style_class = "trace-step-router"
                        icon = "🔀"
                    elif node_name == "rag":
                        style_class = "trace-step-rag"
                        icon = "📚"
                    elif node_name in ("tool_caller", "tools"):
                        style_class = "trace-step-tool"
                        icon = "🔧"
                    else:
                        style_class = ""
                        icon = "⚡"

                    trace_html = f'<div class="trace-step {style_class}">{icon} <b>{node_name}</b>'

                    if "messages" in state_update:
                        messages = state_update["messages"]
                        if isinstance(messages, list) and len(messages) > 0:
                            last_msg = messages[-1]
                            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                                if hasattr(last_msg, "content") and last_msg.content:
                                    # Hiển thị quá trình reasoning (thought)
                                    thought = (
                                        last_msg.content.replace("<thinking>", "")
                                        .replace("</thinking>", "")
                                        .strip()
                                    )
                                    if thought:
                                        trace_html += f"<br>🤔 <i>{thought}</i>"
                                for tc in last_msg.tool_calls:
                                    trace_html += f"<br>→ Tool: <code>{tc['name']}</code>"
                                    used_tools.add(tc["name"])
                            elif hasattr(last_msg, "name") and last_msg.name:
                                content_preview = (
                                    last_msg.content[:100] + "..."
                                    if len(last_msg.content) > 100
                                    else last_msg.content
                                )
                                trace_html += f"<br>→ Result: <code>{content_preview}</code>"
                                import html

                                safe_raw = html.escape(last_msg.content)
                                trace_html += f'<details><summary><b>Trace Step: RawResponse</b></summary><pre style="font-size:11px; max-height:200px; overflow-y:auto; background:#1e1e1e; padding:8px; border-radius:4px;">{safe_raw}</pre></details>'
                                if last_msg.name == "search_documents":
                                    current_sources.append(last_msg.content)

                    if state_update.get("retrieved_context"):
                        ctx_len = len(state_update["retrieved_context"])
                        trace_html += f"<br>→ Retrieved: <code>{ctx_len:,} chars</code>"
                        import html

                        safe_raw_ctx = html.escape(state_update["retrieved_context"])
                        trace_html += f'<details><summary><b>Trace Step: RawResponse</b></summary><pre style="font-size:11px; max-height:200px; overflow-y:auto; background:#1e1e1e; padding:8px; border-radius:4px;">{safe_raw_ctx}</pre></details>'
                        current_sources.append(state_update["retrieved_context"])

                    if "next" in state_update:
                        trace_html += f"<br>→ Route: <code>{state_update['next']}</code>"

                    trace_html += "</div>"
                    current_traces.append(trace_html)
                    status.markdown(trace_html, unsafe_allow_html=True)

            status.update(
                label="✅ **Quá trình Reasoning hoàn tất**", state="complete", expanded=False
            )

            # Get final answer
            import re

            final_state = app.get_state(config)
            answer = final_state.values["messages"][-1].content

            # Extract thinking from the final answer
            thinking_match = re.search(r"<thinking>(.*?)</thinking>", answer, re.DOTALL)
            final_reasoning = ""
            if thinking_match:
                final_reasoning = thinking_match.group(1).strip()
                if final_reasoning:
                    with st.expander(
                        "🤔 **Final Reasoning** (Giải thích lựa chọn dữ liệu)", expanded=True
                    ):
                        st.info(final_reasoning)
                    current_traces.append(
                        f'<div class="trace-step">🤔 <b>Final Reasoning</b><br><i>{final_reasoning}</i></div>'
                    )
                answer = re.sub(r"<thinking>.*?</thinking>", "", answer, flags=re.DOTALL).strip()

            st.markdown(answer)

            if current_sources:
                with st.popover("📚 Nhấn để xem tài liệu trích xuất (Sources)"):
                    for i, doc in enumerate(current_sources):
                        chunks = doc.split("\n\n---\n\n")
                        valid_chunks = []
                        for idx, chunk in enumerate(chunks):
                            chunk = chunk.strip()
                            if not chunk:
                                continue

                            lines = chunk.split("\n")
                            import re

                            title_line = lines[0]
                            match = re.match(r"^\[(\d+)\] Source:\s*(.+)$", title_line)
                            if match:
                                chunk_idx = match.group(1)
                                filename = match.group(2).strip()
                                title = f"[{chunk_idx}] {filename}"
                                content = "\n".join(lines[1:])
                                full_text = answer + "\n" + final_reasoning
                                if f"[{chunk_idx}]" not in full_text and filename not in full_text:
                                    continue
                            else:
                                title = (
                                    title_line
                                    if title_line.startswith("[Source:")
                                    else f"Chunk {idx + 1}"
                                )
                                content = (
                                    "\n".join(lines[1:])
                                    if title_line.startswith("[Source:")
                                    else chunk
                                )
                                if title.startswith("[Source:"):
                                    filename = (
                                        title.replace("[Source:", "").replace("]", "").strip()
                                    )
                                    full_text = answer + "\n" + final_reasoning
                                    if filename not in full_text:
                                        continue

                            snippet = get_relevant_snippet(content, answer)
                            valid_chunks.append((title, snippet))

                        if valid_chunks:
                            st.markdown(f"**Tài liệu được trích xuất (Lần {i + 1}):**")
                            for title, content in valid_chunks:
                                with st.expander(f"📄 {title}"):
                                    st.markdown(content)

            # Prepare summary HTML
            tools_html = " ".join(
                [
                    f'<span style="background:#4CAF50; color:white; padding:2px 6px; border-radius:10px; font-size:10px; margin-right:4px;">{t}</span>'
                    for t in sorted(used_tools)
                ]
            )

            # Save trace
            if tools_html:
                summary_block = f'<div style="margin-top:8px; border-top:1px solid #444; padding-top:4px;"><b>Tools Summary:</b><br>{tools_html}</div>'
            else:
                summary_block = ""

            query_trace = f'<div class="trace-card"><b>Q{st.session_state.total_queries}:</b> {prompt[:50]}...<br>{"".join(current_traces)}{summary_block}</div>'
            st.session_state.traces.append(query_trace)

            # Save message with sources and reasoning
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": current_sources,
                    "reasoning": final_reasoning,
                }
            )

    st.rerun()
