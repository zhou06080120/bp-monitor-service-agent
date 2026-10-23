"""SkillManager 单元测试：CRUD、版本历史、回滚（进化闭环的执行器）。

重点回归场景（对应项目演进中踩过的坑）：
  - patch 回滚必须通过 prev_content 精确恢复，而不是按历史条数删
  - 种子 Skill（无历史文件）被 patch 后回滚，不允许丢失种子内容
  - create 回滚 = 整体移除（目录 + 历史 + 快照）
"""

from skill_manager import SkillManager


def _make_sm(tmp_path) -> SkillManager:
    """构造指向临时目录的 SkillManager（不污染真实 skills/outputs）。"""
    return SkillManager(str(tmp_path / "skills"), str(tmp_path / "versions"))


def _seed_skill(sm: SkillManager, name: str, content: str):
    """模拟一个从未走 create 流程的种子 Skill（直接落盘，无历史文件）。"""
    skill_dir = sm.skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")


# ── create ─────────────────────────────────────────────────────────────────────

def test_create_persists_skill_and_history(tmp_path):
    sm = _make_sm(tmp_path)
    ok = sm.create("device-basics", "---\nversion: 1\n---\n# 设备基础\n袖带绑在左臂。", reason="初始导入")

    assert ok is True
    assert sm.get("device-basics") is not None
    assert "袖带" in sm.load_all()["device-basics"]

    history = sm.get_version_history("device-basics")
    assert len(history) == 1
    assert history[0]["action"] == "create"
    assert history[0]["version"] == 1

    # 快照文件落盘（UI diff 用）
    snap = sm.snapshots_dir / "device-basics_v1.md"
    assert snap.exists()


def test_create_duplicate_returns_false(tmp_path):
    sm = _make_sm(tmp_path)
    sm.create("faq", "内容 A")
    assert sm.create("faq", "内容 B") is False
    # 原内容不被覆盖
    assert sm.get("faq") == "内容 A"


# ── patch ──────────────────────────────────────────────────────────────────────

def test_patch_replaces_first_occurrence_only(tmp_path):
    sm = _make_sm(tmp_path)
    sm.create("faq", "AAAA ---- AAAA")

    ok = sm.patch("faq", "AAAA", "BBBB", reason="替换第一处")
    assert ok is True
    # 仅第一处被替换（patch 的精确性承诺）
    assert sm.get("faq") == "BBBB ---- AAAA"


def test_patch_missing_target_returns_false(tmp_path):
    sm = _make_sm(tmp_path)
    sm.create("faq", "内容")
    assert sm.patch("faq", "不存在的文本", "新内容") is False
    # 失败时原文件不被改动
    assert sm.get("faq") == "内容"


def test_patch_nonexistent_skill_returns_false(tmp_path):
    sm = _make_sm(tmp_path)
    assert sm.patch("ghost", "old", "new") is False


def test_patch_bumps_frontmatter_version(tmp_path):
    sm = _make_sm(tmp_path)
    sm.create("faq", "---\nname: faq\nversion: 1\n---\n正文")
    sm.patch("faq", "正文", "新正文")

    content = sm.get("faq")
    assert "version: 2" in content
    assert "新正文" in content


def test_patch_records_prev_content_for_rollback(tmp_path):
    sm = _make_sm(tmp_path)
    sm.create("faq", "原始内容")
    sm.patch("faq", "原始内容", "进化内容", reason="badcase 修复")

    history = sm.get_version_history("faq")
    assert len(history) == 2
    # prev_content 是 rollback 的唯一依据，必须完整保留
    assert history[1]["prev_content"] == "原始内容"


# ── rollback ───────────────────────────────────────────────────────────────────

def test_rollback_patch_restores_prev_content(tmp_path):
    sm = _make_sm(tmp_path)
    sm.create("faq", "原始内容")
    sm.patch("faq", "原始内容", "进化内容")

    ok = sm.rollback("faq", reason="指标退化")
    assert ok is True
    assert sm.get("faq") == "原始内容"

    # 历史弹出最近一条，版本数减一
    assert len(sm.get_version_history("faq")) == 1


def test_rollback_create_removes_everything(tmp_path):
    sm = _make_sm(tmp_path)
    sm.create("new-skill", "新知识")
    assert sm.get("new-skill") is not None

    ok = sm.rollback("new-skill")
    assert ok is True
    # create 回滚 = 整体移除：正文、历史、快照
    assert sm.get("new-skill") is None
    assert sm.get_version_history("new-skill") == []
    assert not (sm.versions_dir / "new-skill_history.json").exists()
    assert list(sm.snapshots_dir.glob("new-skill_v*.md")) == []


def test_rollback_seed_skill_after_patch_keeps_seed(tmp_path):
    """回归测试：种子 Skill（无历史）被 patch 后回滚，必须完整恢复种子内容。

    历史版本：按"历史条数"回退曾误删初始种子 Skill；现实现按最近一次
    动作类型分派 + prev_content 精确恢复，此用例锁定该行为。
    """
    sm = _make_sm(tmp_path)
    seed = "---\nname: device-basics\nversion: 1\n---\n# 设备基础\n袖带测量要点。"
    _seed_skill(sm, "device-basics", seed)

    assert sm.patch("device-basics", "袖带测量要点。", "袖带测量要点（更新版）。", reason="进化")
    assert sm.rollback("device-basics", reason="回退进化")

    # 种子内容逐字恢复，不允许任何丢失
    assert sm.get("device-basics") == seed


def test_rollback_without_history_returns_false(tmp_path):
    sm = _make_sm(tmp_path)
    _seed_skill(sm, "seed", "种子内容")
    # 从未被 patch/create 过：无历史可回滚
    assert sm.rollback("seed") is False
    assert sm.get("seed") == "种子内容"


def test_rollback_nonexistent_skill_returns_false(tmp_path):
    sm = _make_sm(tmp_path)
    assert sm.rollback("ghost") is False


# ── 版本目录自愈 ───────────────────────────────────────────────────────────────

def test_patch_recreates_deleted_version_dirs(tmp_path):
    """回归测试：版本/快照目录被外部清理后，写入操作需自动重建目录。

    历史版本：目录缺失曾导致进化动作静默落盘失败（全部丢失）。
    """
    sm = _make_sm(tmp_path)
    sm.create("faq", "内容 v1")

    # 模拟外部清理：删掉两个输出目录
    import shutil
    shutil.rmtree(sm.versions_dir)
    shutil.rmtree(sm.snapshots_dir)

    assert sm.patch("faq", "内容 v1", "内容 v2") is True
    assert sm.get("faq") == "内容 v2"
    # 目录被清空后历史从零重建：本次 patch 落为 v1（create 的旧历史已随目录删除）
    history = sm.get_version_history("faq")
    assert len(history) == 1
    assert history[0]["action"] == "patch"
    assert history[0]["content"] == "内容 v2"
