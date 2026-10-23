"""
Skill 变更人工审批 gate —— P4 平台安全核心

Reviewer 产出的 create/patch 动作默认自动落盘（auto，向后兼容）；
BP_APPROVAL_MODE=manual 时转为人工审批流：
  1. Worker 侧在自动应用前调用 request_approval(action_type, payload)：
     返回 False → 本轮跳过落盘，动作进 pending 队列（事件流发 approval_pending）
  2. 管理员经 GET /approvals 查看待审单据
  3. POST /approvals/{id}/approve → 取出 payload 复用 Worker 同款应用函数真正落盘
     POST /approvals/{id}/reject → 仅标记拒绝（不落盘任何变更）

pending 队列：<项目根>/outputs/approvals/pending.jsonl（追加式 JSONL；
状态流转 pending → approved/rejected 时整表重写）。读写全程持线程锁，
后台 Worker 线程与管理端点并发访问安全。
"""

import json        # pending 队列 JSONL 编解码
import logging     # 转待审批/审批决策时的结构化日志
import os          # 读 BP_APPROVAL_MODE 配置
import threading   # pending 文件读写互斥锁
import uuid        # 审批单 id 生成
from datetime import datetime  # 单据时间戳
from pathlib import Path       # 定位项目根下的 pending 队列文件

logger = logging.getLogger(__name__)  # 本模块日志器

ROOT = Path(__file__).resolve().parent.parent  # 项目根目录（本文件在 src/ 下）
PENDING_FILE = ROOT / "outputs" / "approvals" / "pending.jsonl"  # 待审批队列（JSONL）
_lock = threading.Lock()  # 串行化 pending 队列读写（Worker 线程 × 管理端点并发）


def approval_mode() -> str:
    """当前审批模式：BP_APPROVAL_MODE（auto=自动应用【默认】| manual=人工审批）。"""
    return os.getenv("BP_APPROVAL_MODE", "auto").strip().lower()


def request_approval(action_type: str, payload: dict) -> bool:
    """请求放行一条 Skill 变更动作（Worker 自动应用前调用）。

    auto 模式：直接放行（返回 True，调用方照常落盘，向后兼容）；
    manual 模式：动作追加写入 pending 队列并返回 False（调用方跳过应用，
    并在事件流/日志记录"已转入待审批"）。"""
    if approval_mode() != "manual":  # auto（默认）或未知值：保持自动应用
        return True
    entry = {  # 一张审批单
        "id": uuid.uuid4().hex[:12],  # 单据 id（审批端点路由参数）
        "action_type": action_type,   # 动作类型（skill_create / skill_patch）
        "payload": payload,           # 原始动作（approve 时按它真正落盘）
        "ts": datetime.now().isoformat(timespec="seconds"),  # 提审时间
        "status": "pending",          # 状态：pending → approved / rejected
    }
    with _lock:  # 持锁写入：建目录 + 追加单行 JSON 原子完成
        PENDING_FILE.parent.mkdir(parents=True, exist_ok=True)
        with PENDING_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info("Skill 变更已转入待审批: %s/%s", action_type, entry["id"])
    return False


def _load_entries() -> list[dict]:
    """读回全部审批单（调用方须持锁；坏行容错跳过，不中断整个队列）。"""
    if not PENDING_FILE.exists():  # 从未有过待审批动作
        return []
    entries = []  # 结果列表
    for line in PENDING_FILE.read_text(encoding="utf-8").splitlines():  # 逐行解析
        line = line.strip()  # 去首尾空白
        if not line:  # 空行跳过
            continue
        try:
            entries.append(json.loads(line))  # 解析为单据 dict
        except json.JSONDecodeError:  # 坏行（理论上不会出现）
            continue  # 容错跳过
    return entries


def _save_entries(entries: list[dict]) -> None:
    """整表重写 pending 队列（调用方须持锁；状态流转的唯一写路径）。"""
    PENDING_FILE.parent.mkdir(parents=True, exist_ok=True)  # 目录缺失自愈
    with PENDING_FILE.open("w", encoding="utf-8") as f:
        for e in entries:  # 逐单写回
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def _decide(aid: str, status: str) -> dict | None:
    """把指定待审单置为终态并返回该单据（不存在/已处理返回 None；全程持锁）。"""
    with _lock:
        entries = _load_entries()  # 读回全表
        for e in entries:  # 查找目标单据
            if e.get("id") == aid and e.get("status") == "pending":  # 只处理待审单
                e["status"] = status  # 置终态（approved / rejected）
                e["decided_ts"] = datetime.now().isoformat(timespec="seconds")  # 决策时间
                _save_entries(entries)  # 整表落回
                return e  # 返回该单据
    return None  # 未找到可处理单据


def list_pending() -> list[dict]:
    """返回全部待审批单据（供管理端点 GET /approvals 展示）。"""
    with _lock:
        return [e for e in _load_entries() if e.get("status") == "pending"]


def get_pending(aid: str) -> dict | None:
    """按 id 查单据（任意状态可查；不存在返回 None）。"""
    with _lock:
        for e in _load_entries():  # 线性查找（单据量极小，无需索引）
            if e.get("id") == aid:  # 命中
                return e
    return None


def approve(aid: str) -> dict | None:
    """批准单据：状态置 approved 并返回其 payload（调用方据此真正应用）。

    单据不存在或已处理时返回 None（端点据此回 404）。"""
    entry = _decide(aid, "approved")  # 置终态
    if entry is None:  # 无可处理单据
        return None
    logger.info("审批通过: %s (%s)", aid, entry.get("action_type"))  # 决策日志
    return entry.get("payload") or {}  # 返回 payload（缺失时兜底空表）


def reject(aid: str) -> dict | None:
    """拒绝单据：状态置 rejected（不应用任何变更），返回该单据。

    单据不存在或已处理时返回 None（端点据此回 404）。"""
    entry = _decide(aid, "rejected")  # 置终态
    if entry is not None:  # 拒绝成功
        logger.info("审批拒绝: %s (%s)", aid, entry.get("action_type"))  # 决策日志
    return entry  # 返回该单据（供端点回显/审计）
