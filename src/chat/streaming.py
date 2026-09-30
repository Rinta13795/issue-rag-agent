"""Decode only the answer JSON string while model chunks are still arriving."""
import json
import re
import time

from langchain_core.messages import message_chunk_to_message


def partial_answer(text: str) -> str:
    match = re.search(r'\{\s*"answer"\s*:\s*"', text)
    if match is None:
        return ""
    start = match.end()
    i = start
    while i < len(text):
        if text[i] == '"':
            break
        if text[i] == "\\":
            if i + 1 >= len(text):
                break
            if text[i + 1] == 'u':
                if i + 6 > len(text):
                    break
                i += 6
            else:
                i += 2
        else:
            i += 1
    try:
        answer = json.loads('"' + text[start:i] + '"')
    except ValueError:
        return ""
    # Wait for the other half of an escaped Unicode surrogate pair.
    if answer and 0xD800 <= ord(answer[-1]) <= 0xDBFF:
        answer = answer[:-1]
    return answer[:8000]


def streamed_response(model, messages, on_answer):
    if not callable(getattr(model, "stream", None)):
        return model.invoke(messages)
    aggregate = None
    last_answer, last_sent = "", 0.0
    for chunk in model.stream(messages):
        aggregate = chunk if aggregate is None else aggregate + chunk
        # Tool arguments and reasoning metadata never go to the answer display.
        answer = "" if aggregate.tool_call_chunks else partial_answer(aggregate.content) if isinstance(aggregate.content, str) else ""
        now = time.monotonic()
        if answer != last_answer and now - last_sent >= 0.08:
            on_answer(answer)
            last_answer, last_sent = answer, now
    if aggregate is None:
        raise ValueError("模型没有返回流式内容")
    answer = "" if aggregate.tool_call_chunks else partial_answer(aggregate.content) if isinstance(aggregate.content, str) else ""
    if answer != last_answer:
        on_answer(answer)
    return message_chunk_to_message(aggregate)
