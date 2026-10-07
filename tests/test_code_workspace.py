"""本地路径直接读取，无连接和写入能力。"""
from pathlib import Path
import pytest
from src.chat.code_workspace import CodeWorkspace

@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "main.py").write_text("value = 1\nprint(value)\n")
    (tmp_path / ".env").write_text("FAKE_TEST_SECRET=do-not-read")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "hidden.js").write_text("do-not-read")
    return CodeWorkspace(str(tmp_path))

def test_lists_searches_and_reads_without_connection(workspace):
    assert workspace.list_files()["files"] == ["main.py"]
    assert workspace.search_code("VALUE")["matches"][0] == {"path": "main.py", "line": 1, "text": "value = 1"}
    read = workspace.read_file("main.py", 2, 2)
    assert read["content"] == "2: print(value)" and read["total_lines"] == 2
    assert not hasattr(workspace, "edit_file") and not hasattr(workspace, "run_checks")

@pytest.mark.parametrize("path", ["../outside.py", "/tmp/outside.py", ".env", "node_modules/hidden.js", ".git/config", ".local/memory.txt"])
def test_blocks_escape_secrets_and_internal_files(workspace, path):
    with pytest.raises(ValueError): workspace.read_file(path)

def test_blocks_symlink_and_symlink_parent(workspace, tmp_path):
    outside = tmp_path.parent / "outside.py"
    outside.write_text("private = True")
    (tmp_path / "link.py").symlink_to(outside)
    (tmp_path / "linked").symlink_to(tmp_path.parent, target_is_directory=True)
    assert "link.py" not in workspace.list_files()["files"]
    with pytest.raises(ValueError): workspace.read_file("link.py")
    with pytest.raises(ValueError): workspace.read_file("linked/outside.py")

def test_large_binary_and_line_budget(workspace):
    (workspace.root / "binary.py").write_bytes(b"a\0b")
    with pytest.raises(ValueError): workspace.read_file("binary.py")
    with pytest.raises(ValueError): workspace.read_file("main.py", 1, 301)

def test_path_is_not_restricted_to_one_catalog(tmp_path):
    for name in ["mine", "someone-else"]:
        root = tmp_path / name
        root.mkdir()
        (root / "main.py").write_text(name)
        assert name in CodeWorkspace(str(root)).read_file("main.py")["content"]
    with pytest.raises(ValueError): CodeWorkspace("relative")
    with pytest.raises(ValueError): CodeWorkspace(str(tmp_path / "missing"))
    protected = tmp_path / ".aws"
    protected.mkdir()
    with pytest.raises(ValueError): CodeWorkspace(str(protected))

def test_default_root_needs_no_connection(monkeypatch, tmp_path):
    monkeypatch.setattr("src.chat.code_workspace.PROJECT_ROOT", tmp_path)
    assert CodeWorkspace().root == tmp_path
