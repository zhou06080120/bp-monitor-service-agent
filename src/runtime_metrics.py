"""
运行时商业指标聚合：把 business_metrics.json 里的"声明口径"变成"实测口径"。

数据来源（都是实验运行中真实产生的）：
  - 每题回答：耗时（latency）、是否正确（resolved）、是否推脱转人工（deferred）
  - Skill 创建事件：新类目（Skill）从"出现失败样本"到"自动创建并生效"的周期
  - 基线与最终评估准确率：进化增益

派生指标（snapshot()）：
  - deflection_rate_proxy     分流率 = 正确自答 / 总提问（无需人工比例的实测代理）
  - avg_latency_ms / latency_reduction_vs_baseline
     与 business_metrics.json 中"基线约 8.5s"对齐计算的实测降幅
  - category_onboarding       本次实验中自动上线的 Skill 数与其上线耗时

持久化到 outputs/runtime_metrics.json（跨进程可见，/state 读取展示）；
JS 前端直接渲染，无需手工填数。
"""

import json  # 指标 JSON 的读写与序列化
import threading  # 线程锁：并发问答/建技能时保护内存数据结构
from datetime import datetime  # 记录指标更新时间与 Skill 上线时刻
from pathlib import Path  # 定位 outputs/runtime_metrics.json

ROOT = Path(__file__).parent.parent  # 项目根目录（本文件在 src/ 下）
METRICS_FILE = ROOT / "outputs" / "runtime_metrics.json"  # 实测指标持久化文件

# 与 data/business_metrics.json 的 latency baseline 保持一致（人工/排队约 8.5s）
LATENCY_BASELINE_MS = 8500.0  # 人工客服基线延迟（毫秒）：计算"延迟降幅"的分母


# ═════════════════════════════════════════════════════════════════
# RuntimeMetrics —— 商业指标实测聚合器（闭环的"价值证明"出口）
# 流水线每答一题/每建一个 Skill 都调用对应 record_*；snapshot()
# 派生分流率、平均延迟、延迟降幅、Skill 上线数，供前端审计面板渲染
# ═════════════════════════════════════════════════════════════════
class RuntimeMetrics:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else METRICS_FILE  # 持久化文件路径（可注入自定义路径）
        self._lock = threading.Lock()  # 串行化并发读写
        self.data = self._load()  # 启动时读回上次累计的指标（跨进程/重启续算）

    def _load(self) -> dict:
        """读取已有的运行时指标 JSON；不存在或损坏时返回零值结构。"""
        if self.path.exists():  # 有历史指标文件才尝试加载
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))  # 解析历史累计数据
            except json.JSONDecodeError:  # 文件损坏
                pass  # 静默降级为重新累计
        return {"total_questions": 0, "resolved": 0, "deferred": 0,  # 零值结构：总提问/正确/推脱计数
                "latency_ms": 0.0, "latency_count": 0, "latency_min": None,  # 延迟累加值/样本数/最小值
                "latency_max": None, "answers": [],  # 延迟最大值；逐题明细（可选扩展）
                "category_onboarding": [],  # Skill 自动上线事件列表
                "evolution": {"baseline_acc": None, "final_acc": None},  # 基线/最终准确率对照
                "updated_at": None}  # 最近一次落盘时间

    def save(self):
        """把内存中的指标落盘（跨进程可见，/state 据此渲染）。"""
        with self._lock:  # 加锁防止并发写文件
            self.path.parent.mkdir(parents=True, exist_ok=True)  # 确保 outputs/ 目录存在
            self.data["updated_at"] = datetime.now().isoformat()  # 刷新更新时间戳
            self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2),  # 中文不转义、缩进 2 空格
                                 encoding="utf-8")  # UTF-8 写盘

    def record_answer(self, *, question: str, answer: str, correct: bool,
                      deferred: bool, latency_ms: float, run_id: str):
        """累计一次问答的实测指标：总数、正确数、推脱数、延迟统计。"""
        with self._lock:  # 加锁：多线程任务队列会并发调用
            d = self.data  # 快捷引用内存数据
            d["total_questions"] += 1  # 总提问数 +1
            d["resolved"] += int(correct)  # 正确自答数 +1（分流率分子）
            d["deferred"] += int(deferred)  # 推脱转人工数 +1（bool 转 0/1）
            d["latency_ms"] += latency_ms  # 延迟总和累加（算平均值用）
            d["latency_count"] += 1  # 延迟样本数 +1
            d["latency_min"] = latency_ms if d["latency_min"] is None \
                else min(d["latency_min"], latency_ms)  # 维护历史最小延迟
            d["latency_max"] = latency_ms if d["latency_max"] is None \
                else max(d["latency_max"], latency_ms)  # 维护历史最大延迟

    def record_skill_created(self, skill_name: str, run_id: str):
        """记录一个新类目（Skill）自动上线的时刻，用于统计上线周期。"""
        with self._lock:  # 加锁保护列表追加
            self.data["category_onboarding"].append(  # 追加一条上线事件
                {"skill": skill_name, "run_id": run_id,  # 技能名与触发所在运行批次
                 "onboarded_at": datetime.now().isoformat()})  # 上线时间戳（与首次失败样本时刻相减即上线耗时）

    def record_evolution(self, baseline_acc: float | None, final_acc: float | None):
        """记录基线/最终评估准确率，作为进化增益的实测对照。"""
        with self._lock:  # 加锁写入
            self.data["evolution"]["baseline_acc"] = baseline_acc  # 基线准确率（如 33%）
            self.data["evolution"]["final_acc"] = final_acc  # 最终准确率（如 93%）

    def snapshot(self) -> dict:
        """派生指标快照：分流率代理、平均延迟、延迟降幅、Skill 上线数。"""
        with self._lock:  # 加锁读取，避免读到写一半的数据
            d = self.data  # 快捷引用
            n = d["total_questions"] or 1  # 分母：总提问数（0 时取 1 防除零）
            avg = d["latency_ms"] / d["latency_count"] if d["latency_count"] else None  # 平均延迟（无样本则 None）
            reduction = (1 - avg / LATENCY_BASELINE_MS) if avg else None  # 相对人工基线 8.5s 的延迟降幅
            return {  # 组装对外快照（前端直接渲染）
                "total_questions": d["total_questions"],  # 总提问数
                "resolved": d["resolved"],  # 正确自答数
                "deferred": d["deferred"],  # 推脱转人工数
                "deflection_rate_proxy": round(d["resolved"] / n, 3),  # 分流率代理 = 正确自答/总提问
                "avg_latency_ms": round(avg, 1) if avg else None,  # 平均延迟（毫秒，1 位小数）
                "latency_reduction_vs_baseline": round(reduction, 3) if reduction else None,  # 延迟降幅（0~1，相对 8.5s 基线）
                "category_onboarding_count": len(d["category_onboarding"]),  # 本次实验自动上线的 Skill 总数
                "category_onboarding": [  # 最近 10 条上线事件
                    {"skill": x["skill"], "onboarded_at": x["onboarded_at"]}  # 只透出技能名与时间（省略 run_id）
                    for x in d["category_onboarding"][-10:]],  # 取列表末尾 10 条（最新的）
                "evolution": dict(d["evolution"]),  # 基线→最终准确率对照（浅拷贝防外部篡改）
                "updated_at": d["updated_at"],  # 指标最后更新时间
            }
