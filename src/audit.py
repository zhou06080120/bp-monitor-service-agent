"""
管理操作审计留痕 —— P4 平台合规基建

追加式 JSONL 审计日志（不可变写的只增文件）：
  - 默认路径 <项目根>/logs/audit.jsonl，环境变量 BP_AUDIT_PATH 可覆盖
  - 每条记录：ts / actor（Key 指纹，脱敏不落明文）/ action / target / detail / request_id
  - 线程安全：进程内互斥锁串行追加（多进程部署时按文件追加写也是原子单行）

设计口径（三仓同款，同步策略与 auth.py / rate_limit.py 一致）：
  - 谁在何时对什么对象做了什么管理操作（配置变更 / 灰度发布 / 审批 / 重置）
  - actor 来自 auth 中间件写入的 request.state.actor_id（未鉴权时为 "-"）
  - record_audit() 同时向结构化日志输出一条 INFO，便于日志平台检索
"""

import json        # JSON 行式编码
import logging     # 结构化日志同步输出
import os          # 读 BP_AUDIT_PATH 配置
import threading   # 追加写互斥锁
from datetime import datetime, timezone  # 本地时区时间戳
from pathlib import Path                 # 审计文件路径定位

from observability import request_id_var  # 请求编号（跨线程传播的 contextvar）

logger = logging.getLogger(__name__)  # 本模块日志器

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "logs" / "audit.jsonl"  # <项目根>/logs/audit.jsonl
_lock = threading.Lock()             # 串行化追加写，防多线程交错断行


def audit_path() -> Path:
    """当前审计日志路径（BP_AUDIT_PATH 可覆盖，便于测试定向到临时目录）。"""
    return Path(os.getenv("BP_AUDIT_PATH", str(DEFAULT_PATH)))


def record_audit(action: str, target: str = "", detail=None, actor: str = "-") -> dict:
    """追加一条审计记录并返回该条目。

    参数：
      action : 动作名（如 promote_gray / approve_patch / reset_state）
      target : 操作对象标识（如 FAQ id / 审批单 id / 事件序号）
      detail : 附加信息（任意可 JSON 序列化对象，可为 None）
      actor  : 操作者指纹（通常取 request.state.actor_id；未鉴权时保持 "-"）
    """
    entry = {  # 单条审计记录（字段与文档口径一一对应）
        "ts": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),  # 本地时区时间
        "actor": actor,                # 操作者（Key 指纹或 "-"）
        "action": action,              # 动作名
        "target": str(target),         # 对象标识
        "detail": detail,              # 附加信息
        "request_id": request_id_var.get(),  # 关联请求（跨线程传播的上下文）
    }
    path = audit_path()                # 解析目标文件
    with _lock:                        # 全程持锁：建目录 + 追加写原子完成
        path.parent.mkdir(parents=True, exist_ok=True)  # 目录缺失自愈（与 Skill 快照目录自愈同策略）
        with path.open("a", encoding="utf-8") as f:     # 追加模式，只增不改
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")  # 单行 JSON，ensure_ascii 保留中文
    logger.info("审计留痕: %s target=%s", action, target)  # 同步进结构化日志（可检索）
    return entry
