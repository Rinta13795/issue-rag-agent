"""调查工具的有限 GitHub 读取；始终限制在当前仓库。"""
from urllib.parse import urlencode
from src.chat.github_sync import _request_json, fetch_issue, normalize_repository


def read_issue(repository: str, number: int) -> dict:
    repository = normalize_repository(repository)
    issue = fetch_issue(f"https://github.com/{repository}/issues/{number}")
    links, truncated, error = [], False, None
    try:
        for page in range(1, 3):
            events = _request_json(f"https://api.github.com/repos/{repository}/issues/{number}/timeline?per_page=100&page={page}")
            if not isinstance(events, list):
                raise ValueError("无法解析 Issue 时间线")
            for event in events:
                source = event.get("source", {}).get("issue", {})
                url = source.get("html_url", "")
                if source.get("pull_request") and url.lower().startswith(f"https://github.com/{repository}/pull/".lower()):
                    if not any(item["number"] == source.get("number") for item in links):
                        links.append({"number": source["number"], "url": url, "title": source.get("title", ""),
                                      "relationship": "cross_referenced", "event": event.get("event"),
                                      "note": "时间线关联；是否修复该问题需读取 PR 核对"})
            if len(events) < 100:
                break
            truncated = page == 2
    except Exception as exc:
        error = str(exc)[:300]
    return issue.model_copy(update={"linked_prs": links, "links_truncated": truncated, "links_error": error}).model_dump()


def read_pr(repository: str, number: int) -> dict:
    repository = normalize_repository(repository)
    base = f"https://api.github.com/repos/{repository}"
    item = _request_json(f"{base}/pulls/{number}")
    if not isinstance(item, dict) or item.get("number") != number:
        raise ValueError("PR 编号不一致")
    result = {"number": number, "title": str(item.get("title", ""))[:500], "body": str(item.get("body") or "")[:8000],
              "url": item.get("html_url"), "state": item.get("state"), "merged": bool(item.get("merged")),
              "merged_at": item.get("merged_at"), "author": item.get("user", {}).get("login"),
              "body_truncated": len(str(item.get("body") or "")) > 8000}
    for key, path in (("discussion", f"issues/{number}/comments"), ("reviews", f"pulls/{number}/comments"), ("files", f"pulls/{number}/files")):
        try:
            records = _request_json(f"{base}/{path}?per_page=30")
            if not isinstance(records, list):
                raise ValueError("无法解析 PR 附属资料")
            if key == "files":
                budget = 6000
                values = []
                for record in records[:30]:
                    raw = str(record.get("patch") or "")
                    patch = raw[:min(2000, budget)]
                    budget -= len(patch)
                    values.append({"filename": record.get("filename"), "status": record.get("status"),
                                   "patch": patch, "patch_truncated": len(raw) > len(patch) or not raw,
                                   "url": record.get("blob_url")})
                result[key] = values
                result["files_truncated"] = int(item.get("changed_files") or 0) > len(values)
            else:
                result[key] = [{"author": record.get("user", {}).get("login"), "body": str(record.get("body") or "")[:800],
                                "url": record.get("html_url"), "author_association": record.get("author_association")} for record in records[:5]]
                result[key + "_truncated"] = len(records) > 5
        except Exception as exc:
            result[key] = []
            result[key + "_error"] = str(exc)[:300]
    return result


def search_prs(repository: str, query: str) -> dict:
    repository = normalize_repository(repository)
    # Strip scope operators supplied in a query; scope belongs to the program.
    words = [word for word in query.split() if not word.lower().startswith(("repo:", "org:", "user:", "is:", "type:"))]
    params = urlencode({"q": f"repo:{repository} is:pr {' '.join(words)[:180]}", "per_page": 8})
    data = _request_json(f"https://api.github.com/search/issues?{params}")
    return {"relationship": "search_candidate", "note": "相似 PR 候选，不能当成当前 Issue 的关联修复", "items": [
        {"number": item["number"], "title": item.get("title"), "body": str(item.get("body") or "")[:1200], "url": item.get("html_url"), "state": item.get("state")}
        for item in data.get("items", []) if item.get("pull_request")
    ], "incomplete_results": data.get("incomplete_results", False)}
