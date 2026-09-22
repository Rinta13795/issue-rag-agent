"""Demo 可观测运行适配层：线程安全进程内 RunStore 与 SSE 事件发布订阅。"""

import asyncio
import datetime
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Optional

from loguru import logger

from src.demo.models import (
    NodeStatus,
    PipelineNodeInfo,
    QueryAnalysisInfo,
    DecisionInfo,
    RetryRoundInfo,
    RunSnapshot,
    RunStatus,
)

DEFAULT_NODES = [
    ("query_analysis", "01 / QUERY ANALYSIS"),
    ("retrieval", "02 / RETRIEVAL"),
    ("rerank", "03 / RERANK"),
    ("decision", "04 / DECISION"),
]


class RunStore:
    """进程内运行状态仓库：管理 run 快照、节点状态与事件队列。"""

    def __init__(self, max_runs: int = 20, ttl_seconds: int = 3600) -> None:
        self.max_runs = max_runs
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._runs: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._listeners: dict[str, list[asyncio.Queue]] = {}

    def create_run(self, run_id: str, issue_text: str) -> RunSnapshot:
        """创建新 run 并记录到 store。"""
        with self._lock:
            self._cleanup_expired()
            now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
            nodes = [
                PipelineNodeInfo(name=name, label=label, status=NodeStatus.WAITING)
                for name, label in DEFAULT_NODES
            ]
            run_data = {
                "run_id": run_id,
                "status": RunStatus.QUEUED,
                "created_at": now_iso,
                "created_ts": time.time(),
                "started_at": None,
                "completed_at": None,
                "total_elapsed_ms": None,
                "current_node": None,
                "current_round": 1,
                "issue_text": issue_text,
                "nodes": nodes,
                "analysis": None,
                "retrieval": None,
                "rerank": None,
                "decision": None,
                "rounds": [],
                "node_timings_ms": {},
                "error": None,
            }
            if len(self._runs) >= self.max_runs:
                self._runs.popitem(last=False)
            self._runs[run_id] = run_data
            return self._build_snapshot(run_data)

    def get_run(self, run_id: str) -> Optional[RunSnapshot]:
        """获取指定 run 的快照对象。"""
        with self._lock:
            run_data = self._runs.get(run_id)
            if not run_data:
                return None
            return self._build_snapshot(run_data)

    def update_run(self, run_id: str, updater: Callable[[dict[str, Any]], None]) -> Optional[RunSnapshot]:
        """线程安全更新 run 状态，并返回最新快照。"""
        with self._lock:
            run_data = self._runs.get(run_id)
            if not run_data:
                return None
            updater(run_data)
            snapshot = self._build_snapshot(run_data)

        # 触发通知
        return snapshot

    def mark_node_started(self, run_id: str, node_name: str) -> Optional[RunSnapshot]:
        """标记节点开始运行。"""
        with self._lock:
            run_data = self._runs.get(run_id)
            if not run_data:
                return None
            if run_data["status"] == RunStatus.QUEUED:
                run_data["status"] = RunStatus.RUNNING
                run_data["started_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            run_data["current_node"] = node_name
            for node in run_data["nodes"]:
                if node.name == node_name:
                    node.status = NodeStatus.RUNNING
                    break
            snapshot = self._build_snapshot(run_data)
        self.emit_event(run_id, "node.started", {"node": node_name, "run_id": run_id})
        return snapshot

    def mark_node_completed(
        self,
        run_id: str,
        node_name: str,
        elapsed_ms: int,
        summary: Optional[str] = None,
        state_patch: Optional[dict[str, Any]] = None,
    ) -> Optional[RunSnapshot]:
        """标记节点完成。"""
        with self._lock:
            run_data = self._runs.get(run_id)
            if not run_data:
                return None
            run_data["node_timings_ms"][node_name] = elapsed_ms
            for node in run_data["nodes"]:
                if node.name == node_name:
                    node.status = NodeStatus.COMPLETED
                    node.elapsed_ms = elapsed_ms
                    if summary:
                        node.summary = summary
                    break
            if state_patch:
                if "analysis" in state_patch:
                    run_data["analysis"] = state_patch["analysis"]
                if "retrieval" in state_patch:
                    run_data["retrieval"] = state_patch["retrieval"]
                if "rerank" in state_patch:
                    run_data["rerank"] = state_patch["rerank"]
                if "decision" in state_patch:
                    run_data["decision"] = state_patch["decision"]
                if "rounds" in state_patch:
                    run_data["rounds"] = state_patch["rounds"]
                if "current_round" in state_patch:
                    run_data["current_round"] = state_patch["current_round"]
            snapshot = self._build_snapshot(run_data)
        self.emit_event(run_id, "node.completed", {
            "node": node_name,
            "elapsed_ms": elapsed_ms,
            "run_id": run_id,
            "summary": summary,
        })
        return snapshot

    def mark_retry_triggered(self, run_id: str, round_num: int, reason: str) -> Optional[RunSnapshot]:
        """记录触发重试事件并重置节点状态为下一轮。"""
        with self._lock:
            run_data = self._runs.get(run_id)
            if not run_data:
                return None
            run_data["current_round"] = round_num + 1
            for node in run_data["nodes"]:
                node.status = NodeStatus.WAITING
                node.elapsed_ms = None
            snapshot = self._build_snapshot(run_data)
        self.emit_event(run_id, "retry.triggered", {
            "round": round_num,
            "next_round": round_num + 1,
            "reason": reason,
            "run_id": run_id,
        })
        return snapshot

    def mark_completed(self, run_id: str, total_elapsed_ms: int) -> Optional[RunSnapshot]:
        """标记 run 最终完成。"""
        with self._lock:
            run_data = self._runs.get(run_id)
            if not run_data:
                return None
            run_data["status"] = RunStatus.COMPLETED
            run_data["completed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            run_data["total_elapsed_ms"] = total_elapsed_ms
            run_data["current_node"] = None
            snapshot = self._build_snapshot(run_data)
        self.emit_event(run_id, "run.completed", {
            "run_id": run_id,
            "total_elapsed_ms": total_elapsed_ms,
        })
        return snapshot

    def mark_failed(self, run_id: str, error_msg: str) -> Optional[RunSnapshot]:
        """标记 run 失败。"""
        with self._lock:
            run_data = self._runs.get(run_id)
            if not run_data:
                return None
            run_data["status"] = RunStatus.FAILED
            run_data["completed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            run_data["error"] = error_msg
            if run_data.get("current_node"):
                for node in run_data["nodes"]:
                    if node.name == run_data["current_node"]:
                        node.status = NodeStatus.FAILED
            snapshot = self._build_snapshot(run_data)
        self.emit_event(run_id, "run.failed", {
            "run_id": run_id,
            "error": error_msg,
        })
        return snapshot

    def add_listener(self, run_id: str, queue: asyncio.Queue) -> None:
        """添加 SSE 监听队列。"""
        with self._lock:
            if run_id not in self._listeners:
                self._listeners[run_id] = []
            self._listeners[run_id].append(queue)

    def remove_listener(self, run_id: str, queue: asyncio.Queue) -> None:
        """移除 SSE 监听队列。"""
        with self._lock:
            if run_id in self._listeners and queue in self._listeners[run_id]:
                self._listeners[run_id].remove(queue)
                if not self._listeners[run_id]:
                    del self._listeners[run_id]

    def emit_event(self, run_id: str, event_type: str, data: dict[str, Any]) -> None:
        """发布事件给所有监听者。"""
        with self._lock:
            listeners = list(self._listeners.get(run_id, []))
        for q in listeners:
            try:
                q.put_nowait({"event": event_type, "data": data})
            except Exception:
                pass

    def _cleanup_expired(self) -> None:
        """清理超时运行记录。"""
        now = time.time()
        to_delete = []
        for run_id, run_data in self._runs.items():
            if now - run_data.get("created_ts", now) > self.ttl_seconds:
                to_delete.append(run_id)
        for run_id in to_delete:
            del self._runs[run_id]

    def _build_snapshot(self, run_data: dict[str, Any]) -> RunSnapshot:
        """把内部字典转化为不可变快照对象。"""
        return RunSnapshot(
            run_id=run_data["run_id"],
            status=run_data["status"],
            created_at=run_data["created_at"],
            started_at=run_data.get("started_at"),
            completed_at=run_data.get("completed_at"),
            total_elapsed_ms=run_data.get("total_elapsed_ms"),
            current_node=run_data.get("current_node"),
            current_round=run_data.get("current_round", 1),
            issue_text=run_data.get("issue_text", ""),
            nodes=[node.model_copy() for node in run_data["nodes"]],
            analysis=run_data.get("analysis"),
            retrieval=run_data.get("retrieval"),
            rerank=run_data.get("rerank"),
            decision=run_data.get("decision"),
            rounds=[r.model_copy() if hasattr(r, "model_copy") else r for r in run_data.get("rounds", [])],
            node_timings_ms=dict(run_data.get("node_timings_ms", {})),
            error=run_data.get("error"),
        )


_GLOBAL_RUN_STORE = RunStore()


def get_run_store() -> RunStore:
    return _GLOBAL_RUN_STORE
