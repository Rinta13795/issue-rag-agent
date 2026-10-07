"""源码访问边界、版本冲突和实际文件修改的回归。"""
import json
import subprocess
from pathlib import Path

import pytest

from src.chat.code_workspace import CodeWorkspace, workspace_catalog


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("ISSUE_AGENT_WORKSPACES", json.dumps([str(tmp_path)]))
    (tmp_path / "main.py").write_text("value = 1\nprint(value)\n")
    (tmp_path / ".env").write_text("FAKE_TEST_SECRET=do-not-read")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "hidden.js").write_text("do-not-read")
    return CodeWorkspace(workspace_catalog()[0]["id"], allow_edits=True)


def test_lists_searches_and_reads_lines_without_credentials(workspace):
    assert workspace.list_files()["files"] == ["main.py"]
    assert workspace.search_code("VALUE")["matches"][0] == {"path": "main.py", "line": 1, "text": "value = 1"}
    read = workspace.read_file("main.py", 2, 2)
    assert read["content"] == "2: print(value)" and len(read["sha256"]) == 64
    assert read["total_lines"] == 2


@pytest.mark.parametrize("path", ["../outside.py", "/tmp/outside.py", ".env", "node_modules/hidden.js", ".git/config", ".local/memory.txt"])
def test_blocks_escape_secrets_and_internal_files(workspace, path):
    with pytest.raises(ValueError):
        workspace.read_file(path)


def test_blocks_symlink_and_symlink_parent(workspace, tmp_path):
    outside = tmp_path.parent / "outside.py"
    outside.write_text("private = True")
    (tmp_path / "link.py").symlink_to(outside)
    (tmp_path / "linked").symlink_to(tmp_path.parent, target_is_directory=True)
    assert "link.py" not in workspace.list_files()["files"]
    with pytest.raises(ValueError): workspace.read_file("link.py")
    with pytest.raises(ValueError): workspace.edit_file("linked/new.py", "", "x", "missing")


def test_unique_edit_diff_and_stale_version(workspace):
    read = workspace.read_file("main.py")
    result = workspace.edit_file("main.py", "value = 1", "value = 2", read["sha256"])
    assert (workspace.root / "main.py").read_text().startswith("value = 2")
    assert "-value = 1" in result["diff"] and "+value = 2" in result["diff"]
    with pytest.raises(ValueError, match="版本"):
        workspace.edit_file("main.py", "value = 2", "value = 3", read["sha256"])
    assert (workspace.root / "main.py").read_text().startswith("value = 2")


def test_create_and_ambiguous_replacement(workspace):
    workspace.edit_file("tests/test_new.py", "", "assert 1 == 1\n", "missing")
    assert (workspace.root / "tests/test_new.py").read_text() == "assert 1 == 1\n"
    with pytest.raises(ValueError, match="唯一"):
        workspace.edit_file("main.py", "value", "other", workspace.read_file("main.py")["sha256"])


def test_readonly_cannot_write_or_run(workspace):
    readonly = CodeWorkspace(workspace.info["id"])
    with pytest.raises(ValueError, match="只读"):
        readonly.edit_file("main.py", "value = 1", "value = 9", readonly.read_file("main.py")["sha256"])
    with pytest.raises(ValueError, match="开启"):
        readonly.run_checks("python-tests")


def test_bounded_check_preserves_exit_status_and_scrubs_secret(workspace, monkeypatch):
    seen = {}
    monkeypatch.setenv("EXAMPLE_TEST_TOKEN", "fake-token-for-tests")
    class Process:
        returncode = 1
        def wait(self, timeout=None): return 1
    def run(command, **kwargs):
        seen.update(command=command, env=kwargs["env"], cwd=kwargs["cwd"])
        kwargs["stdout"].write(b"FAILED fake-token-for-tests\n" + b"x" * 21000)
        return Process()
    monkeypatch.setattr("src.chat.code_workspace.subprocess.Popen", run)
    result = workspace.run_checks("python-tests", ["main.py"])
    assert result["exit_code"] == 1 and result["truncated"]
    assert "fake-token-for-tests" not in result["output"] and "[redacted]" in result["output"]
    assert "EXAMPLE_TEST_TOKEN" not in seen["env"]
    assert seen["cwd"] == workspace.root and seen["command"][-1] == "main.py"
    with pytest.raises(ValueError): workspace.run_checks("rm -rf")
    with pytest.raises(ValueError): workspace.run_checks("python-tests", ["../outside.py"])


def test_timeout_kills_process_group(workspace, monkeypatch):
    killed = []
    class Process:
        pid = 99999
        returncode = -9
        def wait(self, timeout=None):
            if timeout: raise subprocess.TimeoutExpired("pytest", timeout)
    monkeypatch.setattr("src.chat.code_workspace.subprocess.Popen", lambda *a, **k: Process())
    monkeypatch.setattr("src.chat.code_workspace.os.killpg", lambda pid, sig: killed.append(pid))
    assert workspace.run_checks("python-tests")["timed_out"]
    assert killed == [99999]


def test_large_binary_and_line_budget(workspace):
    (workspace.root / "binary.py").write_bytes(b"a\0b")
    with pytest.raises(ValueError): workspace.read_file("binary.py")
    with pytest.raises(ValueError): workspace.read_file("main.py", 1, 301)


def test_missing_connection_cannot_guess_root():
    with pytest.raises(ValueError, match="选择项目"):
        CodeWorkspace(None)
