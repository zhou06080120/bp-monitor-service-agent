"""
审计留痕 + 高风险人工复核队列。

背景：医疗器械客服的合规要求 ——
  1. 每一次自动应答都要可追溯（问什么、答什么、判对判错、挂了哪些规则分块）
  2. 涉及紧急就医/高危人群的问答，即使"答对了"也要进入人工复核队列，
     不能完全依赖机器裁决。

落盘：
  outputs/audit_log.jsonl    全量问答审计（append-only）
  outputs/risk_queue.jsonl   高风险复核队列（人工在 UI/后台逐条销案）

判定为高风险的规则（确定性、可解释）：
  - 问题涉及紧急医疗词（120/急救/胸痛/晕厥/起搏器…）且机器回答判错；
  - 或回答中出现"120/立即就医/急救"等医疗处置建议（此类建议必须人工复核）。
"""

import json  # 审计条目 JSON 序列化，按行追加写 JSONL
import threading  # 线程锁：并发任务下保证两个 JSONL 文件追加写的原子性
from datetime import datetime  # 生成每条审计的本地时间戳
from pathlib import Path  # 跨平台拼接 outputs/ 下的审计与复核文件路径

ROOT = Path(__file__).parent.parent  # 项目根目录（本文件在 src/ 下，取上一级）
AUDIT_FILE = ROOT / "outputs" / "audit_log.jsonl"  # 全量问答审计日志（每条问答一行，只追加）
RISK_FILE = ROOT / "outputs" / "risk_queue.jsonl"  # 高风险复核队列（人工逐条销案）

# 问题侧急症词表：问题命中且机器判错 → 必进人工复核（急诊不能全信机器）
_EMERGENCY_QUERY_TERMS = ("120", "急救", "胸痛", "晕厥", "昏迷", "呼吸困难",
                          "起搏器", "房颤", "心梗", "脑卒中", "中风")
# 回答侧处置词表：回答出现就医/急救指令 → 无论判对判错都进人工复核
_EMERGENCY_ANSWER_TERMS = ("120", "立即就医", "马上就医", "急救", "拨打急救")


# ═════════════════════════════════════════════════════════════════
# AuditLogger —— 线程安全审计留痕器（闭环的合规出口）
# 每道题评估判分后落一条审计；命中高风险规则则同步写入复核队列，
# pending_risks 供 /state 接口透出到前端"实测指标/审计"面板。
# ═════════════════════════════════════════════════════════════════
class AuditLogger:
    def __init__(self, audit_path: Path | None = None, risk_path: Path | None = None):
        """初始化审计日志与高风险复核队列两个落盘路径（可注入自定义路径）。"""
        self.audit_path = Path(audit_path) if audit_path else AUDIT_FILE  # 审计文件路径（默认全局常量）
        self.risk_path = Path(risk_path) if risk_path else RISK_FILE  # 复核队列路径（默认全局常量）
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)  # 确保 outputs/ 目录存在（含父目录递归创建）
        self._lock = threading.Lock()  # 串行化并发写（inprocess 线程任务队列会并发调用本类）
        self._answers = 0  # 累计审计条数（内存计数，供 stats 汇总）
        self._risks = 0  # 累计高风险标记条数

    def _is_high_risk(self, question: str, answer: str, correct: bool) -> bool:
        """确定性高风险判定：紧急问题答错，或回答中含就医/急救处置建议。"""
        q_emergency = any(t in question for t in _EMERGENCY_QUERY_TERMS)  # 问题是否涉及急症词
        a_advice = any(t in answer for t in _EMERGENCY_ANSWER_TERMS)  # 回答是否含就医/急救处置指令
        return (q_emergency and not correct) or a_advice  # 急症问题答错 或 给出处置建议 → 高风险

    def log_answer(self, *, question: str, answer: str, correct: bool,
                   reason: str = "", run_id: str = "",
                   mount: dict | None = None, judge: dict | None = None,
                   latency_ms: float | None = None):
        """写审计条目；高风险自动入复核队列。返回是否被标记高风险。"""
        risk = self._is_high_risk(question, answer, correct)  # 先做确定性高风险判定（规则词表匹配）
        entry = {  # 组装一条审计记录（长文本字段全部截断防单行爆炸）
            "ts": datetime.now().isoformat(),  # 本地时间戳（ISO 格式，含毫秒）
            "run_id": run_id,  # 所属运行批次（baseline / block_N / final，便于按轮回放）
            "question": question[:200],  # 问题原文截断 200 字
            "answer": answer[:400],  # 回答原文截断 400 字
            "correct": correct,  # 机器判分结果（True/False）
            "reason": reason[:200],  # 判分理由（规则命中词 / 语义分 / Judge 结论）
            "latency_ms": round(latency_ms, 1) if latency_ms is not None else None,  # 单题端到端延迟（毫秒，1 位小数）
            "mount_sections": (mount or {}).get("sections", [])[:8],  # 本次回答实际挂载的规则分块（最多记 8 个）
            "judge": judge or {},  # LLM Judge 原始裁决详情（verdict/reason/votes）
            "risk": risk,  # 高风险标记（True 的条目会同步进复核队列）
        }
        with self._lock:  # 加锁保证并发追加写的行完整性
            with open(self.audit_path, "a", encoding="utf-8") as f:  # 追加模式打开审计文件
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")  # 写一行 JSON（ensure_ascii=False 保留中文）
            self._answers += 1  # 审计计数 +1
            if risk:  # 高风险条目同步写入复核队列
                with open(self.risk_path, "a", encoding="utf-8") as f:  # 追加打开复核队列文件
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")  # 同一条目再落一份到 risk_queue
                self._risks += 1  # 高风险计数 +1
        return risk  # 返回标记结果供调用方记入事件流

    def stats(self) -> dict:
        """返回累计审计条数与高风险标记条数。"""
        with self._lock:  # 加锁读取，避免与并发写竞争
            return {"answers": self._answers, "risk_flagged": self._risks}  # 汇总指标字典（分流率等面板用）

    @staticmethod
    def pending_risks(limit: int = 20) -> list[dict]:
        """读取未销案的复核队列（/state 透出给前端展示）。"""
        if not RISK_FILE.exists():  # 队列文件尚未生成（还没有高风险记录）
            return []  # 返回空列表
        lines = RISK_FILE.read_text(encoding="utf-8").splitlines()[-limit:]  # 只取最近 limit 条（文件持续增长）
        items = []  # 解析结果收集器
        for ln in lines:  # 逐行解析 JSONL
            try:
                items.append(json.loads(ln))  # 解析成功则收集该条目
            except json.JSONDecodeError:  # 半行写入等坏行情况
                continue  # 跳过坏行，不影响其余条目
        return items  # 返回复核条目列表
