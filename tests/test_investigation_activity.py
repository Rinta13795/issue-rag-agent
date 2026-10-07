"""调查统计区别搜索候选、原文读取与失败，不推测已裁掉的历史。"""
from src.chat.models import ChatSession, RuntimeStep
from src.chat.runtime import investigation_activity


def step(tool, status="completed", data=None, args=None, error=None):
    return RuntimeStep(call_id=tool, turn_id="turn", tool=tool, status=status,
                       arguments=args or {}, result={"data": data or {}, **({"error": error} if error else {})})


def test_counts_candidates_separately_from_original_reading():
    session = ChatSession(session_id="s", runtime_steps=[
        step("search_issues", data={"candidates": [{"id": "r:1"}, {"id": "r:2"}]}),
        step("search_issues", data={"candidates": [{"id": "r:2"}, {"id": "r:3"}]}),
        step("read_issue"), step("read_pr", status="failed"),
        step("read_issue", error="读取失败"), step("read_pr", status="cancelled"),
        step("read_project_file", args={"path": "docs/old.md"}),
        step("read_project_file", args={"path": "docs/old.md"}),
        step("read_repository_file"),
    ])
    activity = investigation_activity(session)
    assert activity["issue_candidates_seen"] == 3
    assert activity["issue_reads"] == 1 and activity["pr_reads"] == 0
    assert activity["local_file_reads"] == 2 and activity["local_file_paths"] == ["docs/old.md"]
    assert activity["remote_code_reads"] == 1
    assert activity["successful_tool_results"]["search_issues"] == 2
    assert activity["retained_steps"] == 9


def test_empty_and_bounded_history_do_not_invent_total_reading():
    empty = investigation_activity(ChatSession(session_id="empty"))
    assert empty["issue_candidates_seen"] == empty["issue_reads"] == empty["local_file_reads"] == 0
    session = ChatSession(session_id="s", runtime_steps=[step("read_project_file", args={"path": f"src/{n}.py"}) for n in range(20)])
    activity = investigation_activity(session)
    assert activity["local_file_reads"] == 20
    assert len(activity["local_file_paths"]) == 12 and activity["local_paths_truncated"]
