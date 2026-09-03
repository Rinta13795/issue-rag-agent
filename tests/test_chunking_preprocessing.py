"""文本清洗与递归切分的回归测试。"""

from config import CHUNK_SIZE
from src.data_loader import clean
from src.indexer import extract_error_log, chunk_issue


def _clean_body(body: str, title: str = "Example issue") -> str:
    issue = {
        "id": "issue-1",
        "title": title,
        "body": body,
        "labels": [],
        "component": None,
        "status": "open",
        "resolution": None,
        "project": "demo",
    }
    return clean([issue])[0]["body"]


def _issue_for_chunking(body: str) -> dict:
    return {
        "id": "issue-1",
        "title": "Chunking regression",
        "body": body,
        "labels": [],
        "component": None,
        "status": "open",
        "project": "demo",
    }


def test_body_cleaning_preserves_lines_and_paragraphs() -> None:
    cleaned = _clean_body("para1 line1\npara1 line2\n\npara2")

    assert cleaned == "para1 line1\npara1 line2\n\npara2"


def test_body_cleaning_collapses_only_inline_whitespace() -> None:
    cleaned = _clean_body("alpha   beta\t\tgamma\n delta   epsilon")

    assert cleaned == "alpha beta gamma\ndelta epsilon"


def test_body_cleaning_normalizes_crlf_and_excess_blank_lines() -> None:
    cleaned = _clean_body("first\r\nsecond\r\n\r\n\r\n\rthird")

    assert "\r" not in cleaned
    assert cleaned == "first\nsecond\n\nthird"


def test_title_cleaning_remains_single_line() -> None:
    issue = _issue_for_chunking("body")
    issue["title"] = "  Login\n\tfails   after upgrade  "

    cleaned = clean([issue])[0]

    assert cleaned["title"] == "Login fails after upgrade"


def test_recursive_splitter_prefers_paragraph_boundaries() -> None:
    paragraph_one = "PARA_ONE " + "alpha " * 55
    paragraph_two = "PARA_TWO " + "beta " * 65
    body = _clean_body(f"{paragraph_one}\n\n{paragraph_two}")

    body_chunks = [
        doc.page_content
        for doc in chunk_issue(_issue_for_chunking(body))
        if doc.metadata["chunk_type"] == "body"
    ]

    assert len(body_chunks) >= 2
    assert all(len(chunk) <= CHUNK_SIZE for chunk in body_chunks)
    assert not any("PARA_ONE" in chunk and "PARA_TWO" in chunk for chunk in body_chunks)


def test_recursive_splitter_has_character_fallback() -> None:
    body = "x" * (CHUNK_SIZE * 2 + 137)

    body_chunks = [
        doc.page_content
        for doc in chunk_issue(_issue_for_chunking(body))
        if doc.metadata["chunk_type"] == "body"
    ]

    assert len(body_chunks) >= 3
    assert all(len(chunk) <= CHUNK_SIZE for chunk in body_chunks)


def test_error_log_detection_keeps_multiline_structure() -> None:
    cleaned = _clean_body(
        "Steps before failure\r\n"
        "Traceback (most recent call last):\r\n"
        "  at auth.py:123\r\n"
        "ValueError: invalid token"
    )

    extracted = extract_error_log(cleaned)

    assert extracted.startswith("Traceback (most recent call last):")
    assert "\nat auth.py:123\n" in extracted
    assert extracted.endswith("ValueError: invalid token")
