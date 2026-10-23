"""管理操作审计留痕单元测试：record_audit → JSONL 落盘 + /reset 端点集成。

审计文件经 BP_AUDIT_PATH 指向临时目录，不污染真实 logs/audit.jsonl；
/reset 集成用例把还原/落盘/组件全部替换为假件（对齐 test_api.py 手法，
不触发 lifespan、不触碰真实 outputs/ 现场）。
"""

import json  # 断言 JSONL 可解析

from fastapi.testclient import TestClient  # 内存内 HTTP 客户端

from audit import record_audit  # 被测入口
import serve  # /reset 集成用例对象


# ── record_audit：JSONL 落盘语义 ───────────────────────────────────────────────

def test_record_audit_writes_parseable_entry(monkeypatch, tmp_path):
    """单条留痕：文件存在、单行、六字段齐全、JSON 可解析、返回值与落盘一致。"""
    audit_file = tmp_path / "audit.jsonl"
    monkeypatch.setenv("BP_AUDIT_PATH", str(audit_file))

    entry = record_audit("reset_state", target="experiment",
                         detail={"phase": "idle"}, actor="ab12cd34")

    assert audit_file.exists()  # 文件已落盘
    lines = audit_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1  # 单条单行
    row = json.loads(lines[0])  # JSON 可解析
    for field in ("ts", "actor", "action", "target", "detail", "request_id"):  # 六字段齐全
        assert field in row
    assert row["action"] == "reset_state"  # 动作名原样
    assert row["actor"] == "ab12cd34"  # 操作者指纹原样
    assert row["detail"] == {"phase": "idle"}  # 附加信息原样
    assert entry["action"] == "reset_state"  # 返回值与落盘内容一致


def test_record_audit_appends_monkeypatched_path(monkeypatch, tmp_path):
    """多条留痕追加写（只增不改）：两次调用两行，互不覆盖。"""
    audit_file = tmp_path / "audit.jsonl"
    monkeypatch.setenv("BP_AUDIT_PATH", str(audit_file))

    record_audit("approve_patch", target="a1", actor="user1")
    record_audit("reject_patch", target="b2", actor="user2")

    lines = audit_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2  # 追加而非覆盖
    first, second = (json.loads(ln) for ln in lines)
    assert first["action"] == "approve_patch" and first["target"] == "a1"
    assert second["action"] == "reject_patch" and second["target"] == "b2"


def test_record_audit_defaults(monkeypatch, tmp_path):
    """缺省参数：actor 兜底 "-"、target 兜底空串、detail 允许 None。"""
    audit_file = tmp_path / "audit.jsonl"
    monkeypatch.setenv("BP_AUDIT_PATH", str(audit_file))

    row = record_audit("approve_patch")

    lines = audit_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    saved = json.loads(lines[0])
    assert saved["actor"] == "-"  # 未鉴权操作者兜底
    assert saved["target"] == ""  # target 缺省空串
    assert saved["detail"] is None  # detail 允许为空
    assert row["actor"] == "-"


# ── /reset 端点集成：管理操作必须审计留痕 ────────────────────────────────────

def test_reset_endpoint_writes_audit(monkeypatch, tmp_path):
    """/reset 成功 → 审计日志追加一条 action=reset_state。"""
    audit_file = tmp_path / "audit.jsonl"
    monkeypatch.setenv("BP_AUDIT_PATH", str(audit_file))
    serve._active_job.clear()  # 隔离全局防重表（对齐 test_api.py 的清理手法）
    monkeypatch.setattr(serve, "_restore", lambda: None)  # 还原动作打桩（不碰真实 skills/）
    monkeypatch.setattr(serve, "save_state", lambda s: None)  # 状态落盘打桩（不碰真实 outputs/）

    class _FakeSM:  # 最小技能管理器桩（只用到 load_all）
        @staticmethod
        def load_all():
            return {}

    class _FakeStore:  # 最小索引桩（只用到 reindex_all）
        @staticmethod
        def reindex_all(skills):
            return None

    monkeypatch.setattr(serve, "get_components",
                        lambda: {"sm": _FakeSM(), "store": _FakeStore()})

    client = TestClient(serve.app)  # 不用 with：跳过 lifespan 组件预热
    assert client.post("/reset").status_code == 200

    rows = [json.loads(ln) for ln in audit_file.read_text(encoding="utf-8").strip().splitlines()]
    assert rows[-1]["action"] == "reset_state"  # 最后一条审计即本次重置
    assert rows[-1]["actor"] == "-"  # 测试未配置 Key：actor 兜底
    assert rows[-1]["request_id"]  # 请求编号已关联（observability 中间件注入）
