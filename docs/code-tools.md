# 源码调查与修改工具

2026-10-07 增加。原对话 Runtime 只有当前仓库的 Issue/PR 工具，无法读取用户本地源码或寻找外部实现。此次扩展直接给 Runtime 添加工具，保留原检索、会话、证据与经验流程；没有新增独立 Coding Agent 或 MCP Server。

## 使用

启动本机服务后，新对话选择“读取本地项目”，选择项目并连接。原来的 Issue 对话也可以展开“连接本地项目”选择源码目录。资料仓库与本地源码项目是两个明确的上下文。

直接描述任务，例如“检查检索流程中值得改进的地方，参考公开项目的做法”。Agent 可先列文件、搜索函数名和读取源码，再寻找参考仓库、读取其文件树和少量源码。公开仓库源码使用固定 commit，结果包含仓库、路径、行号和来源链接。Issue/PR 读取也支持显式公开参考仓库；外部 Issue 不覆盖当前资料仓库的焦点。

如需直接修复，在项目入口勾选“允许修改并运行检查”，随后发送具体修改要求。文件工具必须先取得读取返回的整文件 sha256，并唯一匹配原片段；版本冲突时拒绝写入，要求重新读取。创建新文件须使用 `expected_sha256=missing`。没有删除、Git 提交、推送或发布工具。

“查看调查过程”展示实际修改 diff、检查输出、退出码与错误。超时、非零退出码和截断结果不能当作通过。现有会话持久化包含项目连接、修改权限、工具结果和证据。

## 项目配置

默认只配置服务所在项目。新增或替换项目须由启动服务的人设置环境变量，而不是让模型提交任意文件路径：

```bash
export ISSUE_AGENT_WORKSPACES='["/absolute/path/to/project", "/absolute/path/to/another-project"]'
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

源码工作台接口只接受 loopback 客户端、localhost/loopback Host 和本机网页来源。不是远程多用户服务。

## 范围

- 本地工具：`list_project_files`、`search_project_code`、`read_project_file`、`edit_project_file`、`run_project_checks`。
- 参考工具：`search_reference_repositories`、`read_repository_tree`、`read_repository_file`；既有 `read_issue/read_pr/search_prs` 可指定公开 repository。
- 本地文件只支持普通 UTF-8 文本，拒绝越界、软链接、凭据文件、内部状态及依赖目录；单文件最多 250 KB，单次最多读取 300 行。文件列表和搜索有扫描上限，大型项目须缩小到子目录。
- 运行检查限定 `python-tests`、`frontend-build`、`frontend-lint`、`frontend-test`；Python 可指定测试文件，前端检查使用项目 `frontend/package.json` 中相应脚本。项目没有脚本或依赖时会报告失败。
- 检查最多运行 90 秒，超时终止进程组；输出最多 20 KB。检查环境不继承常见凭据变量，输出隐藏已知服务端凭据值。项目测试仍会执行本机代码，必须信任连接的项目；这些限制不构成操作系统沙箱。
- 已连接源码的调查最多 12 次模型调用、24 次工具调用，其余保持原来的 6 次模型、10 次工具预算。最后一次用于总结。修改、检查和本地源码读取不复用旧工具缓存。
- MCP、Hook、Plugin 打包没有在此次实现。后续可复用独立的源码和参考读取模块。

## 验证

新增回归覆盖目录隔离、软链接、凭据排除、只读权限、版本冲突、唯一替换、新文件创建、修改后重新读取、检查退出码/输出截断/超时、公开参考固定 commit、本地会话不依赖 Issue 索引、来源归属和本机接口边界。离线集成使用真实临时文件和模型替身；不代表真实模型一定能选择正确的改进方案。
