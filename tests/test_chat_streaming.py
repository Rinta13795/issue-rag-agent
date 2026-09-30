"""离线检查流式正文、工具协议与重连快照；不调用模型或 GitHub。"""
import asyncio
import json

import pytest
from langchain_core.messages import AIMessageChunk

from src.chat.memory import MemoryStore
from src.chat.service import ChatService
from src.chat.store import ChatStore
from src.chat.streaming import partial_answer, streamed_response


@pytest.mark.parametrize(('raw', 'expected'), [
    ('说明\n```json\n{"answer":"正在查', '正在查'),
    ('{"answer":"第一行\\n第二行\\', '第一行\n第二行'),
    ('{"answer":"引用\\"文字\\"","citations":[]}', '引用"文字"'),
    ('{"answer":"中文\\u4e', '中文'),
    ('{"answer":"\\ud83d', ''),
    ('{"answer":"\\ud83d\\ude00', '😀'),
    ('{"query":"secret"}', ''),
])
def test_partial_answer_decodes_only_complete_characters(raw, expected):
    assert partial_answer(raw) == expected


class StreamingModel:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = 0

    def bind_tools(self, tools):
        return self

    def stream(self, messages):
        self.calls += 1
        yield from self.chunks


def test_tool_chunks_reassemble_without_exposing_arguments():
    model = StreamingModel([
        AIMessageChunk(content='', tool_call_chunks=[{'name': 'read_pr', 'args': '{"number":', 'id': 'c1', 'index': 0}]),
        AIMessageChunk(content='', tool_call_chunks=[{'name': None, 'args': '359}', 'id': None, 'index': 0}]),
    ])
    previews = []
    response = streamed_response(model, [], previews.append)
    assert response.tool_calls[0]['args'] == {'number': 359}
    assert previews == [] and model.calls == 1


def test_runtime_preview_is_transient_and_final_answer_saved_once():
    store = ChatStore()
    session = store.create('repo')
    memory = MemoryStore()
    memory.set_auto_capture(False)
    observed = []
    class Model(StreamingModel):
        def stream(self, messages):
            self.calls += 1
            yield AIMessageChunk(content='{"answer":"调查')
            snapshot = store.stream_snapshot(session.session_id, -1)[0]
            observed.append((snapshot.streaming_answer, len(snapshot.messages)))
            yield AIMessageChunk(content='完成","citations":["fake"]}')
    model = Model([])
    service = ChatService(store=store, answer_llm=model, memory_store=memory)
    mid, _ = store.add_user_message(session.session_id, '你好', 'one')
    service.process_turn(session.session_id, mid)
    result = store.get(session.session_id)
    assert observed == [('调查', 1)]
    assert result.status == 'completed'
    assert result.streaming_answer == '' and result.streaming_turn_id is None
    assert result.messages[-1].content == '调查完成'
    assert result.messages[-1].citations == []
    assert len(result.messages) == 2 and model.calls == 1


def test_preview_does_not_write_database_even_on_other_updates(tmp_path):
    db = tmp_path / 'chat.db'
    store = ChatStore(db_path=db)
    sid = store.create().session_id
    mid, _ = store.add_user_message(sid, '问题', 'one')
    store.update(sid, lambda s: (setattr(s, 'streaming_answer', '未完成'), setattr(s, 'streaming_turn_id', mid)), persist=False)
    store.update(sid, lambda s: setattr(s, 'prompt_tokens', 1))
    payload = json.loads(store._db.execute('SELECT payload FROM chat_sessions').fetchone()[0])
    assert 'streaming_answer' not in payload
    restored = ChatStore(db_path=db).get(sid)
    assert restored.status == 'failed' and restored.streaming_answer == ''
    assert len(restored.messages) == 1


def test_sse_reconnect_receives_full_current_preview_then_final(monkeypatch):
    import api
    store = ChatStore()
    sid = store.create().session_id
    mid, _ = store.add_user_message(sid, '问题', 'one')
    store.update(sid, lambda s: (setattr(s, 'streaming_answer', '已生成'), setattr(s, 'streaming_turn_id', mid)), persist=False)
    monkeypatch.setattr(api, 'get_chat_store', lambda: store)
    class Request:
        async def is_disconnected(self):
            return False
    async def read():
        response = await api.chat_session_events(sid, Request())
        iterator = response.body_iterator
        first = await anext(iterator)
        assert 'event: snapshot' in first and '已生成' in first
        store.update(sid, lambda s: setattr(s, 'streaming_answer', '已生成更多'), persist=False)
        second = await anext(iterator)
        assert 'event: answer' in second and '已生成更多' in second
        store.update(sid, lambda s: (setattr(s, 'status', 'completed'), setattr(s, 'streaming_answer', '')))
        third = await anext(iterator)
        assert 'event: snapshot' in third and '"status": "completed"' in third
        await iterator.aclose()
    asyncio.run(read())
