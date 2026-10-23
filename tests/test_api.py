"""API 层单元测试：阶段门禁（409）、任务提交回执、状态查询。

不触发 lifespan（TestClient 不用 with）：组件预热、Skill 备份等重操作
不在单测范围内；load_state / tm / get_components 均被替换为受控假件，
保证用例不依赖、不污染真实的 outputs/experiment_state.json。
"""

from fastapi.testclient import TestClient
import pytest

import serve


client = TestClient(serve.app)  # 不用 with：跳过 lifespan 组件预热


@pytest.fixture(autouse=True)
def _clear_active_jobs():
    """隔离 serve 模块级 _active_job：防止用例间互相污染（防重表是全局状态）。"""
    serve._active_job.clear()
    yield
    serve._active_job.clear()


def _state(phase: str = "idle", current_block: int = 0) -> dict:
    """构造最小合法实验状态（serve._submit 只读 phase / current_block）。"""
    return {"phase": phase, "current_block": current_block,
            "nudge_count": 0, "eval_results": {}, "conv_history": []}


class FakeTaskManager:
    """记录提交调用、永不真正执行的假队列。"""

    def __init__(self):
        self.calls = []

    def submit(self, task, params=None):
        self.calls.append((task, params))
        return "fake-job-001"

    def get(self, job_id):
        return {"id": job_id, "status": "done"}

    def poll_events(self, job_id, after_seq):
        return [], self.get(job_id)


def _fake_components(blocks: int = 8) -> dict:
    return {"blocks": [[{"block": f"block-{i}"}] for i in range(blocks)]}


# ── 阶段门禁：非法状态必须 409 ─────────────────────────────────────────────────

def test_baseline_rejected_when_not_idle(monkeypatch):
    monkeypatch.setattr(serve, "load_state", lambda: _state("complete"))
    r = client.post("/job/baseline")
    assert r.status_code == 409


def test_final_rejected_when_blocks_incomplete(monkeypatch):
    monkeypatch.setattr(serve, "load_state", lambda: _state("baseline_done"))
    r = client.post("/job/final")
    assert r.status_code == 409


def test_block_rejected_on_wrong_index(monkeypatch):
    monkeypatch.setattr(serve, "load_state", lambda: _state("baseline_done", current_block=0))
    monkeypatch.setattr(serve, "get_components", lambda: _fake_components())
    # 状态指针在 0，提交第 3 块 → 跳块拒绝
    r = client.post("/job/block/3")
    assert r.status_code == 409


def test_block_rejected_on_out_of_range(monkeypatch):
    monkeypatch.setattr(serve, "load_state", lambda: _state("baseline_done", current_block=8))
    monkeypatch.setattr(serve, "get_components", lambda: _fake_components())
    # 状态指针已到 8（越界），提交第 8 块 → 拒绝
    r = client.post("/job/block/8")
    assert r.status_code == 409


def test_block_submit_on_correct_index(monkeypatch):
    monkeypatch.setattr(serve, "load_state", lambda: _state("baseline_done", current_block=0))
    monkeypatch.setattr(serve, "get_components", lambda: _fake_components())
    fake = FakeTaskManager()
    monkeypatch.setattr(serve, "tm", fake)
    r = client.post("/job/block/0")
    assert r.status_code == 200
    assert fake.calls == [("block", {"block_index": 0})]


# ── 提交回执：合法状态 200 + job_id ────────────────────────────────────────────

def test_baseline_submit_returns_job_id(monkeypatch):
    monkeypatch.setattr(serve, "load_state", lambda: _state("idle"))
    fake = FakeTaskManager()
    monkeypatch.setattr(serve, "tm", fake)

    r = client.post("/job/baseline")
    assert r.status_code == 200
    body = r.json()
    assert body["job_id"] == "fake-job-001"
    assert body["task"] == "baseline"
    # serve._submit 内部 params or {} 归一化：无参任务以空字典入队
    assert fake.calls == [("baseline", {})]


def test_final_submit_after_all_blocks(monkeypatch):
    monkeypatch.setattr(serve, "load_state", lambda: _state("all_blocks_done", current_block=8))
    fake = FakeTaskManager()
    monkeypatch.setattr(serve, "tm", fake)

    r = client.post("/job/final")
    assert r.status_code == 200
    assert fake.calls == [("final", {})]


def test_duplicate_task_key_rejected(monkeypatch):
    """同类任务运行中重复提交 → 409（_active_job 防重）。"""
    monkeypatch.setattr(serve, "load_state", lambda: _state("idle"))

    class RunningTM(FakeTaskManager):
        def get(self, job_id):
            return {"id": job_id, "status": "running"}  # 永远运行中

    fake = RunningTM()
    monkeypatch.setattr(serve, "tm", fake)

    assert client.post("/job/baseline").status_code == 200
    assert client.post("/job/baseline").status_code == 409


# ── 查询接口 ───────────────────────────────────────────────────────────────────

def test_job_active_empty_when_no_running_jobs():
    r = client.get("/job/active")
    assert r.status_code == 200
    assert r.json() == {}


def test_job_status_unknown_id():
    r = client.get("/job/nonexistent-id")
    assert r.status_code == 200
    assert r.json()["status"] == "unknown"
