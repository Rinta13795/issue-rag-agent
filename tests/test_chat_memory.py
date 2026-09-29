"""经验记忆独立于短期会话；只有确认后写入，且按仓库与问题取用。"""

from src.chat.memory import MemoryStore
from src.chat.models import MemoryProposal


def test_confirmed_memory_survives_restart_and_can_be_forgotten(tmp_path):
    path = tmp_path / "memories.sqlite3"
    store = MemoryStore(path)
    preference = store.add(MemoryProposal(kind="preference", scope="global", text="提 Issue 前先查重", source_excerpt="先查重"),
                           repository_id="repo-a", session_id="chat-1")
    experience = store.add(MemoryProposal(kind="experience", scope="repository", text="登录 401 与旧 Issue 12 有关，用户已核对", source_excerpt="登录 401"),
                           repository_id="repo-a", session_id="chat-1")
    restored = MemoryStore(path)
    assert {item.memory_id for item in restored.list("repo-a")} == {preference.memory_id, experience.memory_id}
    assert [item.memory_id for item in restored.select("repo-b", "登录 401")] == [preference.memory_id]
    assert experience.memory_id in [item.memory_id for item in restored.select("repo-a", "登录 401 无法使用")]
    assert restored.delete(preference.memory_id)
    assert preference.memory_id not in [item.memory_id for item in MemoryStore(path).list()]


def test_repository_experience_requires_repository_id():
    store = MemoryStore()
    try:
        store.add(MemoryProposal(kind="experience", scope="repository", text="确认过的处理经验", source_excerpt="确认过"),
                  repository_id=None, session_id="chat-1")
    except ValueError:
        pass
    else:
        raise AssertionError("仓库经验不能没有仓库归属")
