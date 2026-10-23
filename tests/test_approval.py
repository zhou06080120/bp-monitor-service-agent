"""审批 gate 单元测试：双模式行为 + pending 状态流转 + 原调用点跳过应用 + 审批端点。

pending 队列经 approval.PENDING_FILE 指向临时目录（模块函数读该常量，
monkeypatch 替换即可隔离，不污染真实 outputs/approvals/）；
审批端点用例复用业务 app（TestClient 不进 lifespan，对齐 test_api.py 手法），
get_components 替换为假件、_restore/save_state 打桩，零真实 LLM 调用。
"""

import json  # 断言 JSONL 可解析

import pytest  # fixture/monkeypatch
from fastapi.testclient import TestClient  # 内存内 HTTP 客户端

import approval as approval_mod  # 先导模块：隔离 PENDING_FILE 需直接操作
import pipeline  # 被测的执行链/应用函数
import serve  # 审批端点所在业务 app
from approval import approve, get_pending, list_pending, reject, request_approval  # 被测入口
from skill_manager import SkillManager  # 用真实管理器验证落盘路径


@pytest.fixture(autouse=True)
def _isolate_pending_file(tmp_path, monkeypatch):
    """pending 队列指到临时目录（隔离真实 outputs/approvals/）。"""
    monkeypatch.setattr(approval_mod, "PENDING_FILE", tmp_path / "pending.jsonl")
    yield


class _FakeMetrics:
    """运行时指标桩（execute_reviewer_action 的 create 路径会记账）。"""

    def __init__(self):
        self.created = []  # 记录被记账的新技能名

    def record_skill_created(self, name, run_id):
        self.created.append(name)


def _make_emitter():
    """事件收集器：返回 (emit 回调, 已收集事件列表)。"""
    events = []

    def emit(etype, **payload):
        events.append({"type": etype, **payload})

    return emit, events


_ACT = {"action": "patch", "skill_name": "faq",
        "old_text": "内容 A", "new_text": "内容 B", "reason": "修复"}  # 标准 patch 动作


# ── request_approval：双模式行为 ───────────────────────────────────────────────

def test_auto_mode_allows_immediately(monkeypatch):
    """auto 模式：直接放行（True），不落任何单据（向后兼容）。"""
    monkeypatch.setenv("BP_APPROVAL_MODE", "auto")
    assert request_approval("skill_patch", _ACT) is True
    assert not approval_mod.PENDING_FILE.exists()  # 队列文件根本不产生


def test_default_mode_is_auto(monkeypatch):
    """未配置 BP_APPROVAL_MODE：默认 auto（零改动升级，现有部署行为不变）。"""
    monkeypatch.delenv("BP_APPROVAL_MODE", raising=False)
    assert approval_mod.approval_mode() == "auto"
    assert request_approval("skill_patch", _ACT) is True


def test_manual_mode_defers_and_records(monkeypatch):
    """manual 模式：拦截（False）并把动作追加写入 pending 队列。"""
    monkeypatch.setenv("BP_APPROVAL_MODE", "manual")
    assert request_approval("skill_patch", _ACT) is False  # 调用方据此跳过应用

    lines = approval_mod.PENDING_FILE.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1  # 单条单行
    row = json.loads(lines[0])
    assert row["status"] == "pending"  # 初始状态
    assert row["action_type"] == "skill_patch"
    assert row["payload"] == _ACT  # 原始动作完整入队（approve 时按它落盘）
    assert row["id"] and row["ts"]  # 单据 id 与时间戳齐备


# ── pending 队列：状态流转 ────────────────────────────────────────────────────

def test_lifecycle_approve_returns_payload(monkeypatch):
    """pending → approved：approve 返回 payload，已决单不再出现在待审列表。"""
    monkeypatch.setenv("BP_APPROVAL_MODE", "manual")
    request_approval("skill_patch", _ACT)
    aid = list_pending()[0]["id"]

    assert get_pending(aid)["status"] == "pending"  # 决策前待审
    assert approve(aid) == _ACT  # 批准并取回 payload
    assert get_pending(aid)["status"] == "approved"  # 状态已流转
    assert list_pending() == []  # 已决单退出待审列表
    assert approve(aid) is None  # 重复批准无效（防二次应用）


def test_lifecycle_reject_keeps_payload_unapplied(monkeypatch):
    """pending → rejected：仅标记拒绝，单据保留（可审计追溯）。"""
    monkeypatch.setenv("BP_APPROVAL_MODE", "manual")
    request_approval("skill_patch", _ACT)
    aid = list_pending()[0]["id"]

    entry = reject(aid)
    assert entry is not None and entry["status"] == "rejected"
    assert list_pending() == []
    assert reject(aid) is None  # 重复拒绝无效


def test_unknown_id_decisions_return_none(monkeypatch):
    """不存在的单据 id：approve/reject 均返回 None（端点据此回 404）。"""
    monkeypatch.setenv("BP_APPROVAL_MODE", "manual")
    assert approve("no-such-id") is None
    assert reject("no-such-id") is None


# ── 原调用点：execute_reviewer_action 的审批 gate 包裹 ────────────────────────

def test_manual_mode_skips_apply_at_call_point(monkeypatch, tmp_path):
    """manual 模式：Worker 执行链跳过落盘、发 approval_pending 事件、单据入队。"""
    monkeypatch.setenv("BP_APPROVAL_MODE", "manual")
    sm = SkillManager(str(tmp_path / "skills"), str(tmp_path / "versions"))
    sm.create("faq", "内容 A")  # 种一个技能
    emit, events = _make_emitter()
    executed = []  # run_block 语义下的"已落盘清单"

    pipeline.execute_reviewer_action(sm, _FakeMetrics(), _ACT, emit, 0, executed)

    assert executed == []  # 未落盘（不计入已执行 → 不触发索引重建/回滚闸门）
    assert sm.get("faq") == "内容 A"  # 原文未被改动
    assert any(e["type"] == "approval_pending" for e in events)  # 事件流已留痕
    # 单据已入队；审批通过后经同款应用函数补落盘
    aid = list_pending()[0]["id"]
    assert approve(aid) == _ACT
    assert pipeline.apply_skill_action(sm, _ACT) is True
    assert sm.get("faq") == "内容 B"


def test_auto_mode_applies_at_call_point(monkeypatch, tmp_path):
    """auto 模式：执行链照常落盘（向后兼容），发 skill_action 事件。"""
    monkeypatch.setenv("BP_APPROVAL_MODE", "auto")
    sm = SkillManager(str(tmp_path / "skills"), str(tmp_path / "versions"))
    sm.create("faq", "内容 A")
    emit, events = _make_emitter()
    executed = []

    pipeline.execute_reviewer_action(sm, _FakeMetrics(), _ACT, emit, 0, executed)

    assert executed == [{"action": "patch", "skill_name": "faq", "reason": "修复"}]
    assert sm.get("faq") == "内容 B"  # 已落盘
    assert any(e["type"] == "skill_action" for e in events)
    assert list_pending() == []  # auto 模式不产生单据


def test_redline_blocked_never_enters_approval(monkeypatch, tmp_path):
    """红线阻断优先于审批：危险动作不落盘、也不进待审队列。"""
    monkeypatch.setenv("BP_APPROVAL_MODE", "manual")
    sm = SkillManager(str(tmp_path / "skills"), str(tmp_path / "versions"))
    sm.create("faq", "内容 A")
    emit, events = _make_emitter()
    dangerous = {"action": "patch", "skill_name": "faq", "old_text": "内容 A",
                  "new_text": "起搏器用户可以放心使用电子血压计", "reason": "危险规则"}

    pipeline.execute_reviewer_action(sm, _FakeMetrics(), dangerous, emit, 0, [])

    assert any(e["type"] == "redline_blocked" for e in events)  # 红线事件
    assert list_pending() == []  # 危险动作没有资格进审批队列
    assert sm.get("faq") == "内容 A"  # 原文未被改动


# ── 审批端点（业务 app + 假组件，鉴权未配置=放行） ───────────────────────────

def _fake_components(monkeypatch, sm):
    """把 serve.get_components 替换为最小假件（sm 真实、store 打桩）。"""
    class _FakeStore:
        @staticmethod
        def reindex_all(skills):
            return None

    monkeypatch.setattr(serve, "get_components",
                        lambda: {"sm": sm, "store": _FakeStore()})


def test_list_approvals_endpoint(monkeypatch):
    """GET /approvals：返回待审单据列表。"""
    monkeypatch.setenv("BP_APPROVAL_MODE", "manual")
    request_approval("skill_patch", _ACT)
    client = TestClient(serve.app)

    r = client.get("/approvals")

    assert r.status_code == 200
    pending = r.json()["pending"]
    assert len(pending) == 1
    assert pending[0]["payload"] == _ACT


def test_approve_endpoint_applies_and_audits(monkeypatch, tmp_path):
    """POST /approvals/{id}/approve：payload 经 Worker 同款函数落盘 + 审计留痕。"""
    monkeypatch.setenv("BP_AUDIT_PATH", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("BP_APPROVAL_MODE", "manual")
    sm = SkillManager(str(tmp_path / "skills"), str(tmp_path / "versions"))
    sm.create("faq", "内容 A")
    _fake_components(monkeypatch, sm)
    request_approval("skill_patch", _ACT)
    aid = list_pending()[0]["id"]

    r = TestClient(serve.app).post(f"/approvals/{aid}/approve")

    assert r.status_code == 200
    assert r.json() == {"status": "approved", "applied": True}
    assert sm.get("faq") == "内容 B"  # 与 Worker 自动应用同一条落盘路径
    rows = [json.loads(ln) for ln in
            (tmp_path / "audit.jsonl").read_text(encoding="utf-8").strip().splitlines()]
    assert rows[-1]["action"] == "approve_patch"  # 审计动作名
    assert rows[-1]["target"] == aid  # 审计对象=审批单 id
    assert list_pending() == []  # 单据已决


def test_approve_endpoint_unknown_id_404(monkeypatch, tmp_path):
    """POST /approvals/{未知 id}/approve → 404（不误应用任何东西）。"""
    monkeypatch.setenv("BP_AUDIT_PATH", str(tmp_path / "audit.jsonl"))
    r = TestClient(serve.app).post("/approvals/no-such-id/approve")
    assert r.status_code == 404


def test_reject_endpoint_marks_and_audits(monkeypatch, tmp_path):
    """POST /approvals/{id}/reject：仅标记拒绝 + 审计留痕，Skill 不受影响。"""
    monkeypatch.setenv("BP_AUDIT_PATH", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("BP_APPROVAL_MODE", "manual")
    sm = SkillManager(str(tmp_path / "skills"), str(tmp_path / "versions"))
    sm.create("faq", "内容 A")
    _fake_components(monkeypatch, sm)
    request_approval("skill_patch", _ACT)
    aid = list_pending()[0]["id"]

    r = TestClient(serve.app).post(f"/approvals/{aid}/reject")

    assert r.status_code == 200
    assert r.json() == {"status": "rejected"}
    assert sm.get("faq") == "内容 A"  # 拒绝=不落盘任何变更
    assert get_pending(aid)["status"] == "rejected"
    rows = [json.loads(ln) for ln in
            (tmp_path / "audit.jsonl").read_text(encoding="utf-8").strip().splitlines()]
    assert rows[-1]["action"] == "reject_patch"
    assert rows[-1]["target"] == aid
