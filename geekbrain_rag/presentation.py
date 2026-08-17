import re
from collections import Counter

STOPWORDS = {
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


def relevant_snippet(chunk_text: str, answer: str) -> str:
    """Select a compact excerpt that overlaps with the rendered answer."""
    clean_answer = re.sub(r"[^\w\s]", "", answer).lower()
    keywords = {word for word in clean_answer.split() if word not in STOPWORDS and len(word) > 1}

    if not keywords:
        return chunk_text[:200] + ("..." if len(chunk_text) > 200 else "")

    word_frequency = Counter(re.findall(r"\b\w+\b", chunk_text.lower()))
    lines = chunk_text.splitlines()
    best_line_index = -1
    best_score = -1.0

    for index, line in enumerate(lines):
        if not line.strip():
            continue
        score = sum(
            100.0 / word_frequency.get(keyword, 1)
            for keyword in keywords
            if re.search(r"\b" + re.escape(keyword) + r"\b", line.lower())
        )
        if score > best_score:
            best_score = score
            best_line_index = index

    if best_line_index == -1:
        raw_snippet = chunk_text[:200].strip()
    else:
        start_index = max(0, best_line_index - 1)
        end_index = min(len(lines), best_line_index + 3)
        raw_snippet = "\n".join(lines[start_index:end_index]).strip()

    if not raw_snippet:
        return ""

    trimmed_chunk = chunk_text.strip()
    has_prefix = not trimmed_chunk.startswith(raw_snippet)
    has_suffix = not trimmed_chunk.endswith(raw_snippet)

    prefix = "... " if has_prefix else ""
    suffix = " ..." if has_suffix else ""
    return f"{prefix}{raw_snippet}{suffix}"


def remove_private_reasoning(text: str) -> str:
    """Drop model-only thinking blocks before rendering or storing an answer."""
    return re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL).strip()


def source_items(raw_sources: list[str], answer: str) -> list[tuple[str, str]]:
    """Return the cited, de-duplicated source excerpts used by an answer."""
    items: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

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
                items.append((title, relevant_snippet(content, answer)))

    return items
