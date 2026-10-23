"""
异步任务队列抽象：把耗时的评估 / 进化闭环从 HTTP 请求协程中剥离到后台 Worker。

背景问题：
  原实现用 asyncio.to_thread 把阻塞调用丢进线程池，但每个 SSE 请求内仍是
  "边跑边吐"的长事务 —— 多用户并发触发 Self-Evolution 时：
    · 请求占满事件循环外的线程池，新请求排队等待
    · 全局实验状态被并发长事务互相污染
    · 前端断线后任务即被取消，无法恢复

本模块解耦方案：
  1. 提交即返回：POST /job/xxx 只做校验 + 入队，立刻返回 job_id
  2. 单 Worker 串行执行 Pipeline 任务（实验状态天然互斥，不需要分布式锁）
  3. 事件日志（内存 + JSONL 落盘）承载进度；SSE 端点按 seq 增量拉取推送
  4. 后端可插拔：
       - InProcessBackend（默认）：进程内线程 Worker，零依赖
       - CeleryQueueBackend（可选）：TASK_BACKEND=celery 时启用，
         Worker 可在独立进程/机器跑（见 celery_app.py），
         进度经 outputs/jobs/{job_id}.jsonl 落盘，SSE 轮询同一份日志

两者共用 pipeline.run_task 与同一事件日志格式 → 前端完全无感。
"""

import os  # 读取 TASK_BACKEND 环境变量（inprocess / celery 切换）
import json  # 事件 JSONL 落盘的序列化/反序列化
import queue  # 进程内任务队列（线程安全 FIFO）
import uuid  # 生成全局唯一 job_id
import threading  # 后台 Worker 线程与并发锁
import traceback  # 任务失败时记录异常堆栈尾部
from pathlib import Path  # 定位 outputs/jobs/ 事件日志目录
from datetime import datetime  # 事件时间戳与任务创建时间

ROOT = Path(__file__).parent.parent  # 项目根目录（本文件在 src/ 下）
JOBS_DIR = ROOT / "outputs" / "jobs"  # 每个 job 一份 {job_id}.jsonl 事件日志


# ═════════════════════════════════════════════════════════════════
# EventLog —— 单任务事件流：内存环形缓冲 + JSONL 落盘
# 内存缓冲供进程内 SSE 快速增量拉取（since()）；JSONL 供 Celery
# 跨进程架构下同一端点轮询读取，两种后端共用同一事件格式
# ═════════════════════════════════════════════════════════════════
class EventLog:
    """单 job 的事件序列：内存环形缓冲 + JSONL 落盘（供 Celery 跨进程读取）。"""

    MAX_MEMORY_EVENTS = 20000  # 内存环形缓冲上限（超出裁掉最旧事件，JSONL 仍有全量）

    def __init__(self, job_id: str):
        self.job_id = job_id  # 所属任务 ID
        self.seq = 0  # 全局递增事件序号（SSE 增量拉取的游标）
        self.memory: list[dict] = []          # [{seq, type, ...payload}]
        JOBS_DIR.mkdir(parents=True, exist_ok=True)  # 确保 jobs 目录存在
        self.file = JOBS_DIR / f"{job_id}.jsonl"  # 本任务的落盘日志文件
        self._lock = threading.Lock()  # 保护 seq 与缓冲的并发写

    def append(self, event_type: str, **payload) -> int:
        with self._lock:  # 加锁：Worker 线程写事件时独占
            self.seq += 1  # 序号自增（从 1 开始）
            # 真实序号必须最后写入：业务 payload 里可能带 seq 字段（如题号），
            # 若被其覆盖会破坏 SSE 增量游标（since() 过滤导致事件永久丢失）
            event = {**payload, "seq": self.seq,
                     "ts": datetime.now().isoformat(), "type": event_type}
            self.memory.append(event)  # 追加进内存缓冲
            if len(self.memory) > self.MAX_MEMORY_EVENTS:  # 缓冲超上限
                self.memory = self.memory[-self.MAX_MEMORY_EVENTS:]  # 保留最近 N 条（环形效果）
            with open(self.file, "a", encoding="utf-8") as f:  # 追加打开落盘文件
                f.write(json.dumps(event, ensure_ascii=False) + "\n")  # 同步写一行 JSON（崩溃也不丢已发事件）
            return self.seq  # 返回本事件序号（emit 回调可感知进度）

    def since(self, after_seq: int) -> list[dict]:
        with self._lock:  # 加锁读取，避免与追加写竞争
            return [e for e in self.memory if e["seq"] > after_seq]  # 返回所有序号大于游标的增量事件


# ── In-Process：线程 Worker + 内存事件 ─────────────────────────────────────────

# ═════════════════════════════════════════════════════════════════
# InProcessBackend —— 默认后端：进程内线程 Worker（零依赖）
# 提交即返回 job_id；后台线程从队列取任务串行执行 pipeline.run_task，
# 进度经 EventLog 承载，SSE 端点按 seq 增量拉取
# ═════════════════════════════════════════════════════════════════
class InProcessBackend:
    backend = "inprocess"  # 后端标识（/state 透出，前端展示当前架构形态）

    def __init__(self, worker_count: int = 1):
        self.jobs: dict[str, dict] = {}  # job_id → 任务状态字典
        self.events: dict[str, EventLog] = {}  # job_id → 事件日志实例
        self._q: queue.Queue = queue.Queue()  # 待执行任务的 FIFO 队列
        self._lock = threading.Lock()  # 保护 jobs/events 字典的并发访问
        for i in range(max(1, worker_count)):  # 至少启动 1 个 Worker 线程
            threading.Thread(target=self._worker, name=f"job-worker-{i}",
                             daemon=True).start()  # daemon 线程：主进程退出即结束

    def submit(self, task: str, params: dict | None = None) -> str:
        job_id = uuid.uuid4().hex[:12]  # 生成 12 位十六进制随机任务 ID
        self.jobs[job_id] = {"id": job_id, "task": task, "params": params or {},
                             "status": "pending", "result": None, "error": None,
                             "created": datetime.now().isoformat()}  # 登记任务初始状态（pending 待执行）
        self.events[job_id] = EventLog(job_id)  # 为该任务创建专属事件日志
        self.events[job_id].append("job_queued", task=task)  # 记录入队事件（SSE 首条：已排队）
        self._q.put(job_id)  # 投入执行队列（Worker 线程会立刻取走）
        return job_id  # 立即返回 job_id，HTTP 提交请求到此结束（毫秒级）

    def get(self, job_id: str) -> dict:
        return self.jobs.get(job_id, {"id": job_id, "status": "unknown"})  # 查询任务状态；不存在则 unknown

    def poll_events(self, job_id: str, after_seq: int) -> tuple[list[dict], dict]:
        """返回 (新事件列表, job 状态)。SSE 端点周期调用。"""
        log = self.events.get(job_id)  # 取该任务的事件日志
        events = log.since(after_seq) if log else []  # 拉取游标之后的增量事件（无日志则空）
        return events, self.get(job_id)  # 事件与任务状态一并返回

    # ═══════════════════════════════════════════════════════════
    # _worker：后台线程主循环——从队列取任务 → 执行 pipeline →
    # 记录结果/异常。单 Worker 串行保证实验状态互斥，无需分布式锁
    # ═══════════════════════════════════════════════════════════
    def _worker(self):
        while True:  # 常驻循环（daemon 线程随进程退出）
            job_id = self._q.get()  # 阻塞等待下一个任务（无任务时挂起）
            job = self.jobs.get(job_id)  # 取任务登记信息
            if not job:  # 登记信息丢失（异常情况）
                continue  # 跳过继续等下一个
            job["status"] = "running"  # 标记为执行中
            log = self.events[job_id]  # 取该任务事件日志
            log.append("job_started", task=job["task"], params=job["params"])  # 记录开始事件（含参数快照）
            try:
                from pipeline import run_task  # 延迟导入：避免循环依赖且加快启动
                result = run_task(job["task"], job["params"],
                                  emit=lambda t, **p: log.append(t, **p))  # 执行流水线；进度回调直写事件日志
                job["result"] = result  # 保存最终结果（指标等）
                job["status"] = "done"  # 标记成功
                log.append("job_done", result=result)  # 记录完成事件（前端据此收尾）
            except Exception as e:  # 执行异常
                job["status"] = "failed"  # 标记失败
                job["error"] = str(e)[:500]  # 保存错误摘要
                log.append("job_error", message=str(e)[:500],
                           trace=traceback.format_exc()[-2000:])  # 记录失败事件（错误 + 堆栈尾部 2000 字）
            finally:
                self._q.task_done()  # 通知队列该任务处理完毕（join 场景用）


# ── Celery 后端：跨进程 Worker，进度经 JSONL 落盘 ──────────────────────────────

# ═════════════════════════════════════════════════════════════════
# CeleryQueueBackend —— 可选后端：任务经 Redis 投递到独立 Worker 进程
# Web 进程只负责 submit（投递）与 poll_events（读 JSONL）；
# 执行进度由 Worker 写 outputs/jobs/{job_id}.jsonl，跨进程可见
# ═════════════════════════════════════════════════════════════════
class CeleryQueueBackend:
    backend = "celery"  # 后端标识

    def __init__(self):
        from celery_app import app  # 导入 Celery 应用（此时要求已安装 celery 并起 Redis）
        self.app = app  # 保存应用实例

    def submit(self, task: str, params: dict | None = None) -> str:
        job_id = uuid.uuid4().hex[:12]  # 生成任务 ID（与进程内后端同一格式）
        log_path = JOBS_DIR / f"{job_id}.jsonl"  # 该任务的落盘日志路径
        JOBS_DIR.mkdir(parents=True, exist_ok=True)  # 确保 jobs 目录存在
        log_path.write_text("", encoding="utf-8")  # 预创建空日志文件（poll 端据此可读）
        # 异步投递：Redis broker 排队，Worker 进程执行
        self.app.send_task("pipeline.celery_run_task",
                           args=[job_id, task, params or {}])  # 按任务名投递（Worker 端 celery_run_task 接收）
        return job_id  # 立即返回（真正执行在另一个进程）

    def get(self, job_id: str) -> dict:
        from celery_app import app  # 取 Celery 应用（幂等导入）
        res = app.AsyncResult(f"bp:{job_id}")  # 查询任务结果（Redis result backend）
        return {"id": job_id, "status": res.status.lower(),
                "result": res.result if res.status == "SUCCESS" else None}  # 状态小写统一；仅成功时带结果

    def poll_events(self, job_id: str, after_seq: int) -> tuple[list[dict], dict]:
        """从 JSONL 落盘日志增量读取（跨进程可见）。"""
        log_path = JOBS_DIR / f"{job_id}.jsonl"  # 该任务的日志文件
        events = []  # 收集游标之后的增量事件
        if log_path.exists():  # 日志文件存在才解析
            with open(log_path, encoding="utf-8") as f:  # 打开（Worker 持续追加）
                for line in f:  # 从头逐行读（文件量级小，全量扫描可接受）
                    try:
                        e = json.loads(line)  # 解析单行事件
                    except json.JSONDecodeError:  # 读到半行（Worker 正在写）
                        continue  # 跳过，下次轮询再读
                    if e.get("seq", 0) > after_seq:  # 序号大于游标 → 是新事件
                        events.append(e)  # 收集
        return events, self.get(job_id)  # 与进程内后端返回结构完全一致


# ── 统一门面 ──────────────────────────────────────────────────────────────────

# ═════════════════════════════════════════════════════════════════
# TaskManager —— 统一门面：serve.py 只认识它
# 按 TASK_BACKEND 选择实现，两种后端接口一致 → 前端完全无感
# ═════════════════════════════════════════════════════════════════
class TaskManager:
    """按 TASK_BACKEND 环境变量选择后端（默认 inprocess，零依赖）。"""

    def __init__(self):
        backend_env = os.getenv("TASK_BACKEND", "inprocess").lower()  # 读取后端配置（默认 inprocess）
        self.impl = (CeleryQueueBackend() if backend_env == "celery"
                     else InProcessBackend())  # celery → 跨进程后端；其余一律进程内线程后端

    def submit(self, task: str, params: dict | None = None) -> str:
        return self.impl.submit(task, params)  # 透传提交，返回 job_id

    def get(self, job_id: str) -> dict:
        return self.impl.get(job_id)  # 透传状态查询

    def poll_events(self, job_id: str, after_seq: int) -> tuple[list[dict], dict]:
        return self.impl.poll_events(job_id, after_seq)  # 透传事件增量拉取（SSE 用）
