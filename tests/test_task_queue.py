"""异步任务队列单元测试：EventLog 游标语义 + InProcessBackend 提交执行。

重点回归场景（对应踩过的坑）：
  - 事件真实序号必须最后写入，不能被业务 payload 里的 seq 字段（如题号）
    覆盖——否则 SSE 增量游标失效，前端永久丢事件
"""

import time

import task_queue as tq


# ── EventLog ───────────────────────────────────────────────────────────────────

def _log_in_tmp(monkeypatch, tmp_path):
    """把 JOBS_DIR 指到临时目录，避免污染真实 outputs/jobs/。"""
    monkeypatch.setattr(tq, "JOBS_DIR", tmp_path)
    return tq.EventLog("job-test")


def test_eventlog_seq_increments(monkeypatch, tmp_path):
    log = _log_in_tmp(monkeypatch, tmp_path)
    assert log.append("job_started") == 1
    assert log.append("eval_q") == 2
    assert log.append("eval_complete") == 3


def test_eventlog_seq_not_overwritten_by_payload(monkeypatch, tmp_path):
    """回归测试：业务 payload 携带 seq=999（题号）时，真实序号仍须保持 1。"""
    log = _log_in_tmp(monkeypatch, tmp_path)
    log.append("eval_q", seq=999, question="第 999 题")

    event = log.since(0)[0]
    assert event["seq"] == 1  # 真实事件序号，未被 payload 覆盖
    assert event["question"] == "第 999 题"  # 业务字段原样保留


def test_eventlog_since_returns_only_increment(monkeypatch, tmp_path):
    log = _log_in_tmp(monkeypatch, tmp_path)
    for i in range(5):
        log.append("tick", i=i)

    assert len(log.since(0)) == 5   # 全量
    assert len(log.since(3)) == 2   # 游标 3 之后：seq 4、5
    assert [e["seq"] for e in log.since(3)] == [4, 5]


def test_eventlog_persists_to_jsonl(monkeypatch, tmp_path):
    log = _log_in_tmp(monkeypatch, tmp_path)
    log.append("job_queued", task="baseline")

    # 事件同步落盘（崩溃也不丢已发事件）
    import json
    lines = (tmp_path / "job-test.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["type"] == "job_queued"


# ── InProcessBackend ───────────────────────────────────────────────────────────

def test_inprocess_submit_executes_and_reports(monkeypatch, tmp_path):
    """提交 → 后台线程执行（run_task 被 mock）→ 状态与事件可见。"""
    monkeypatch.setattr(tq, "JOBS_DIR", tmp_path)

    import pipeline
    monkeypatch.setattr(pipeline, "run_task",
                        lambda task, params, emit: (emit("eval_q", q=1),
                                                    emit("eval_complete", acc=1.0),
                                                    {"accuracy": 1.0})[-1])

    backend = tq.InProcessBackend()
    job_id = backend.submit("baseline")

    # 等待后台线程执行完成（单 Worker 串行，正常毫秒级）
    for _ in range(100):
        if backend.get(job_id)["status"] in ("done", "failed"):
            break
        time.sleep(0.05)

    job = backend.get(job_id)
    assert job["status"] == "done"
    assert job["result"] == {"accuracy": 1.0}

    events, job = backend.poll_events(job_id, 0)
    types = [e["type"] for e in events]
    assert "job_queued" in types
    assert "job_started" in types
    assert "eval_q" in types
    assert "job_done" in types


def test_inprocess_task_failure_recorded(monkeypatch, tmp_path):
    """任务抛异常时标记 failed，错误与堆栈写入事件日志（前端可见原因）。"""
    monkeypatch.setattr(tq, "JOBS_DIR", tmp_path)

    import pipeline
    def _boom(task, params, emit):
        raise RuntimeError("评估集文件缺失")
    monkeypatch.setattr(pipeline, "run_task", _boom)

    backend = tq.InProcessBackend()
    job_id = backend.submit("block", {"block_index": 0})

    for _ in range(100):
        if backend.get(job_id)["status"] in ("done", "failed"):
            break
        time.sleep(0.05)

    job = backend.get(job_id)
    assert job["status"] == "failed"
    assert "评估集文件缺失" in job["error"]

    events, _ = backend.poll_events(job_id, 0)
    assert any(e["type"] == "job_error" for e in events)


def test_inprocess_get_unknown_job():
    backend = tq.InProcessBackend()
    assert backend.get("no-such-job")["status"] == "unknown"


# ── TaskManager 门面 ───────────────────────────────────────────────────────────

def test_task_manager_defaults_to_inprocess():
    tm = tq.TaskManager()  # conftest 已固定 TASK_BACKEND=inprocess
    assert tm.impl.backend == "inprocess"
