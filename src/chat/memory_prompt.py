"""A bounded model view; the original job retains provenance and revision guards."""
import json
from copy import deepcopy
from src.chat.context import select_message_excerpt

MEMORY_INPUT_CHAR_BUDGET = 16000


def memory_prompt_payload(payload):
    result = {key: deepcopy(payload.get(key)) for key in (
        'session_id', 'message_id', 'repository_id', 'case_id', 'investigation_summary',
        'user_messages', 'assistant_message', 'source_issue', 'candidates',
    )}
    result['investigation_summary'] = str(result.get('investigation_summary') or '')[:1200]
    result['user_messages'] = (result.get('user_messages') or [])[-3:]
    for item in result['user_messages']:
        item['content'] = select_message_excerpt(str(item.get('content') or ''), 600)
    if result.get('source_issue'):
        result['source_issue']['body'] = select_message_excerpt(str(result['source_issue'].get('body') or ''), 1000)
    result['existing_records'] = [
        {key: deepcopy(item.get(key)) for key in (
            'memory_id', 'revision', 'kind', 'scope', 'status', 'entry_type',
            'case_id', 'text', 'source_excerpt', 'modified_by', 'verification_scope',
        )} for item in (payload.get('existing_records') or [])[:12]
    ]
    for item in result['existing_records']:
        item['text'] = str(item.get('text') or '')[:400]
        item['source_excerpt'] = str(item.get('source_excerpt') or '')[:180]
    # Current-turn tools only. New turns without tools do not resend past PR diffs.
    ids = payload.get('current_tool_evidence_ids')
    evidence = payload.get('tool_evidence') or []
    evidence = [item for item in evidence if item.get('id') in ids] if ids is not None else evidence[-3:]
    result['tool_evidence'] = []
    for item in evidence[:6]:
        text = str(item.get('text') or '')
        result['tool_evidence'].append({
            'id': item.get('id'), 'kind': item.get('kind'), 'title': str(item.get('title') or '')[:240],
            'url': item.get('url'), 'text': select_message_excerpt(text, 1000),
            'metadata': deepcopy(item.get('metadata') or {}),
            'truncated': bool(item.get('truncated')) or len(text) > 1000,
        })
    # Drop optional material, never clip serialized JSON into an invalid object.
    while len(json.dumps(result, ensure_ascii=False, separators=(',', ':'))) > MEMORY_INPUT_CHAR_BUDGET:
        for key in ('existing_records', 'tool_evidence', 'candidates'):
            if result.get(key):
                result[key].pop()
                break
        else:
            raise ValueError('记忆整理的基础输入超过预算')
    return result
