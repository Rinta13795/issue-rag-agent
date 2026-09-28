"""将 HKUDS/OpenHarness 的公开 Issue 同步到独立本地检索库。

运行：python -m scripts.sync_openharness_issues
需要 GitHub API 可访问；可设置 GITHUB_TOKEN 提高 API 额度。不会修改 GitBugs 主索引。
"""

import json
import os
import pickle
import re
import ssl
import time
from datetime import datetime, timezone
from urllib.request import Request, urlopen

from langchain_chroma import Chroma
import certifi
from rank_bm25 import BM25Okapi

from config import INDEX_BATCH_SIZE
from src.chat.repositories import OVERLAY_DIR
from src.docstore import build_docstore
from src.indexer import chunk_issue, load_embeddings, tokenize


def fetch_issues() -> list[dict]:
    url = "https://api.github.com/repos/HKUDS/OpenHarness/issues?state=all&per_page=100"
    records: list[dict] = []
    while url:
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "issue-rag-agent"}
        if os.environ.get("GITHUB_TOKEN"):
            headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
        for attempt in range(3):
            try:
                with urlopen(Request(url, headers=headers), timeout=30,
                             context=ssl.create_default_context(cafile=certifi.where())) as response:
                    page = json.load(response)
                    links = response.headers.get("Link", "")
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        records.extend(item for item in page if "pull_request" not in item)
        match = re.search(r'<([^>]+)>; rel="next"', links)
        url = match.group(1) if match else ""
    return records


def normalize(item: dict) -> dict:
    return {
        "id": f"openharness:{item['number']}",
        "title": item["title"] or "",
        "body": item.get("body") or "",
        "labels": [label["name"] for label in item.get("labels", [])],
        "component": "",
        "status": item.get("state", "open"),
        "project": "openharness",
    }


def main() -> None:
    issues = [normalize(item) for item in fetch_issues()]
    if not issues:
        raise RuntimeError("未获取到 OpenHarness Issue，拒绝创建空索引")
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    # 与 GitBugs 主库分开，不覆盖其 10.6 万 Issue 的索引。
    bm25 = BM25Okapi([tokenize(f"{issue['title']} {issue['body']}") for issue in issues])
    with (OVERLAY_DIR / "bm25.pkl").open("wb") as file:
        pickle.dump({"bm25": bm25, "ids": [issue["id"] for issue in issues]}, file)
    build_docstore(issues, path=str(OVERLAY_DIR / "docstore.pkl"))
    store = Chroma(collection_name="issues", embedding_function=load_embeddings(),
                   persist_directory=str(OVERLAY_DIR / "chroma"))
    batch_docs, batch_ids = [], []
    for issue in issues:
        for index, doc in enumerate(chunk_issue(issue)):
            batch_docs.append(doc)
            batch_ids.append(f"{issue['id']}:{index}")
            if len(batch_docs) >= INDEX_BATCH_SIZE:
                store.add_documents(batch_docs, ids=batch_ids)
                batch_docs, batch_ids = [], []
    if batch_docs:
        store.add_documents(batch_docs, ids=batch_ids)
    manifest = {"repository": "HKUDS/OpenHarness", "issue_count": len(issues),
                "synced_at": datetime.now(timezone.utc).isoformat()}
    (OVERLAY_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OpenHarness: 已同步 {len(issues)} 条 Issue 到 {OVERLAY_DIR}")


if __name__ == "__main__":
    main()
