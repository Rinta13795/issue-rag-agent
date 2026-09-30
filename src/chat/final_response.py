"""Distinguish model response format failures from investigation conclusions."""
import json
import re
from dataclasses import dataclass


@dataclass
class FinalResponse:
    answer: str | None
    citations: list[str]
    mode: str
    error: str | None = None


def parse_final_response(response, allowed: set[str]) -> FinalResponse:
    content = response.content
    if isinstance(content, list):
        content = "\n".join(block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text")
    if not isinstance(content, str) or not content.strip():
        return FinalResponse(None, [], "invalid", "模型没有返回最终回答；此前证据已保留，可以重试本轮。")
    text = content.strip()
    metadata = getattr(response, "response_metadata", None) or {}
    if metadata.get("finish_reason") in ("length", "max_tokens"):
        return FinalResponse(None, [], "truncated", "模型回答达到长度上限，未完整生成；此前证据已保留，可以重试本轮。")
    cleaned = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except (ValueError, TypeError):
        # Accept a complete answer object after introductory text or inside a fence.
        # Never treat a broken structured answer as ordinary prose.
        value = None
        decoder = json.JSONDecoder()
        candidates = list(re.finditer(r'\{\s*"answer"\s*:', text))
        for match in candidates:
            try:
                candidate, _ = decoder.raw_decode(text[match.start():])
            except ValueError:
                continue
            if isinstance(candidate, dict) and isinstance(candidate.get("answer"), str):
                value = candidate
                break
        if value is None:
            if candidates or cleaned.startswith(("{", "[")) or text.lower().startswith("```json"):
                return FinalResponse(None, [], "invalid_json", "模型返回的回答格式不完整；此前证据已保留，可以重试本轮。")
            return FinalResponse(text, [], "plain_text")
    if not isinstance(value, dict) or not isinstance(value.get("answer"), str) or not value["answer"].strip():
        return FinalResponse(None, [], "invalid_schema", "模型返回的结构中缺少有效回答；此前证据已保留，可以重试本轮。")
    citations = value.get("citations")
    valid = [item for item in citations if isinstance(item, str) and item in allowed] if isinstance(citations, list) else []
    return FinalResponse(value["answer"].strip(), list(dict.fromkeys(valid)), "json")


def is_unspecified_resolution(text: str) -> bool:
    """Only short affirmative closure reports; never infer which suggested fix worked."""
    compact = re.sub(r"[\s，。！!,.：:]", "", text).casefold()
    return bool(re.fullmatch(r"(?:ok|okay|好的|好了|嗯|好)?(?:我)?(?:已经|已|现在)?(?:解决了|搞定了|好了|修好了|修复了)", compact))
