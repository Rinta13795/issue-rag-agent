"""Demo 可观测运行执行器：生命周期事件与状态追踪。"""

import concurrent.futures
import time
from typing import Any, Optional

from loguru import logger

from config import CONFIDENCE_THRESHOLD, MAX_RETRIES
from src.agent.graph import _ensure_dependencies, should_retry
from src.agent.nodes import (
    decision_node,
    query_analysis_node,
    rerank_node,
    retrieval_node,
)
from src.agent.state import IssueState
from src.demo.models import (
    CandidateInfo,
    DecisionInfo,
    QueryAnalysisInfo,
    RetryRoundInfo,
    RunSnapshot,
)
from src.demo.run_store import RunStore, get_run_store


class ObservableRunner:
    """可观测运行执行器：使用单 worker 线程池调度任务，逐步追踪节点耗时与快照。"""

    def __init__(self, run_store: Optional[RunStore] = None) -> None:
        self.store = run_store or get_run_store()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag-demo-worker")

    def submit_run(self, run_id: str, issue_text: str) -> None:
        """异步提交运行任务到后台线程池。"""
        self._executor.submit(self._execute_run, run_id, issue_text)

    def _execute_run(self, run_id: str, issue_text: str) -> None:
        """在工作线程中逐步执行 RAG 状态机并发出生命周期事件。"""
        logger.info("开始执行可观测任务 run_id={}", run_id)
        start_time = time.time()
        try:
            _ensure_dependencies()

            state: IssueState = {
                "raw_issue": issue_text,
                "retry_count": 0,
                "previous_decisions": [],
            }

            rounds_history: list[RetryRoundInfo] = []
            current_round = 1

            while True:
                # ----------------- 1. Query Analysis -----------------
                self.store.mark_node_started(run_id, "query_analysis")
                t0 = time.time()
                qa_res = query_analysis_node(state)
                elapsed_qa = int((time.time() - t0) * 1000)
                state.update(qa_res)

                analysis_info = QueryAnalysisInfo(
                    rewritten_query=state.get("rewritten_query", ""),
                    keywords=state.get("keywords", []),
                    component=state.get("component"),
                    component_filter_applied=state.get("component_filter_applied", False),
                    component_filter_note=state.get("component_filter_note"),
                )
                qa_summary = f"Keywords: {len(analysis_info.keywords)}, Component: {analysis_info.component or 'None'}"
                self.store.mark_node_completed(
                    run_id,
                    "query_analysis",
                    elapsed_qa,
                    summary=qa_summary,
                    state_patch={"analysis": analysis_info, "current_round": current_round},
                )

                # ----------------- 2. Retrieval -----------------
                self.store.mark_node_started(run_id, "retrieval")
                t0 = time.time()
                ret_res = retrieval_node(state)
                elapsed_ret = int((time.time() - t0) * 1000)
                state.update(ret_res)

                retrieved_candidates = [
                    CandidateInfo(
                        id=str(doc.get("id", "")),
                        title=str(doc.get("title", "")),
                        body_snippet=str(doc.get("body", ""))[:240],
                        score=round(float(doc.get("score", 0.0)), 4) if doc.get("score") is not None else None,
                    )
                    for doc in state.get("retrieved_docs", [])
                ]
                retrieval_data = {
                    "candidate_count": len(retrieved_candidates),
                    "candidates": retrieved_candidates,
                }
                ret_summary = f"召回 {len(retrieved_candidates)} 条候选"
                # 更新 analysis 里的 component_filter_applied / note
                analysis_info.component_filter_applied = state.get("component_filter_applied", False)
                analysis_info.component_filter_note = state.get("component_filter_note")
                self.store.mark_node_completed(
                    run_id,
                    "retrieval",
                    elapsed_ret,
                    summary=ret_summary,
                    state_patch={"retrieval": retrieval_data, "analysis": analysis_info},
                )

                # ----------------- 3. Rerank -----------------
                self.store.mark_node_started(run_id, "rerank")
                t0 = time.time()
                rerank_res = rerank_node(state)
                elapsed_rerank = int((time.time() - t0) * 1000)
                state.update(rerank_res)

                reranked_candidates = [
                    CandidateInfo(
                        id=str(doc.get("id", "")),
                        title=str(doc.get("title", "")),
                        body_snippet=str(doc.get("body", ""))[:240],
                        score=round(float(doc.get("score", 0.0)), 4) if doc.get("score") is not None else None,
                        rerank_score=round(float(doc.get("rerank_score", 0.0)), 4) if doc.get("rerank_score") is not None else None,
                    )
                    for doc in state.get("reranked_docs", [])
                ]
                rerank_data = {
                    "candidate_count": len(reranked_candidates),
                    "candidates": reranked_candidates,
                }
                rerank_summary = f"精排保留 {len(reranked_candidates)} 条候选"
                self.store.mark_node_completed(
                    run_id,
                    "rerank",
                    elapsed_rerank,
                    summary=rerank_summary,
                    state_patch={"rerank": rerank_data},
                )

                # ----------------- 4. Decision -----------------
                self.store.mark_node_started(run_id, "decision")
                t0 = time.time()
                dec_res = decision_node(state)
                elapsed_dec = int((time.time() - t0) * 1000)
                state.update(dec_res)

                decision_info = DecisionInfo(
                    decision=state["decision"],
                    confidence=round(state["confidence"], 2),
                    related_issues=[str(i) for i in state.get("related_issues", [])],
                    reasoning=state.get("reasoning", ""),
                    retry_count=state.get("retry_count", 1),
                )

                # 标记 candidates 中的 is_related
                related_set = set(decision_info.related_issues)
                for cand in reranked_candidates:
                    cand.is_related = cand.id in related_set
                for cand in retrieved_candidates:
                    cand.is_related = cand.id in related_set

                # 记录本轮
                last_hist = state.get("previous_decisions", [])[-1] if state.get("previous_decisions") else {}
                round_info = RetryRoundInfo(
                    round=current_round,
                    rewritten_query=state.get("rewritten_query", ""),
                    keywords=state.get("keywords", []),
                    component=state.get("component"),
                    decision=state["decision"],
                    confidence=state["confidence"],
                    related_issues=decision_info.related_issues,
                    reasoning=decision_info.reasoning,
                    retrieved_count=len(retrieved_candidates),
                    candidate_count=len(reranked_candidates),
                    top_score=last_hist.get("top_score"),
                    score_gap=last_hist.get("score_gap"),
                )
                rounds_history.append(round_info)

                dec_summary = f"Decision: {decision_info.decision.upper()} (conf={decision_info.confidence:.2f})"
                self.store.mark_node_completed(
                    run_id,
                    "decision",
                    elapsed_dec,
                    summary=dec_summary,
                    state_patch={
                        "decision": decision_info,
                        "rerank": {"candidate_count": len(reranked_candidates), "candidates": reranked_candidates},
                        "retrieval": {"candidate_count": len(retrieved_candidates), "candidates": retrieved_candidates},
                        "rounds": rounds_history,
                    },
                )

                # ----------------- 条件边判断 -----------------
                next_step = should_retry(state)
                if next_step == "retry":
                    reason = f"置信度 {state['confidence']:.2f} 低于阈值 {CONFIDENCE_THRESHOLD}，触发第 {current_round + 1} 轮诊断改写"
                    logger.warning("run_id={} 触发重试: {}", run_id, reason)
                    self.store.mark_retry_triggered(run_id, current_round, reason)
                    current_round += 1
                else:
                    break

            total_elapsed = int((time.time() - start_time) * 1000)
            self.store.mark_completed(run_id, total_elapsed)
            logger.info("可观测任务 run_id={} 完成，总耗时 {} ms", run_id, total_elapsed)

        except Exception as exc:
            logger.exception("可观测任务 run_id={} 异常失败", run_id)
            self.store.mark_failed(run_id, f"{type(exc).__name__}: {str(exc)}")


_GLOBAL_RUNNER: Optional[ObservableRunner] = None


def get_observable_runner() -> ObservableRunner:
    global _GLOBAL_RUNNER
    if _GLOBAL_RUNNER is None:
        _GLOBAL_RUNNER = ObservableRunner()
    return _GLOBAL_RUNNER
