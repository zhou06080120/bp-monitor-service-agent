"""
Celery 入口（可选）：把 Pipeline 任务投递到独立 Worker 进程执行。

启用方式：
  1. pip install celery redis
  2. 起 Redis（broker + backend）
  3. 环境变量：
       TASK_BACKEND=celery
       CELERY_BROKER_URL=redis://localhost:6379/0
       CELERY_RESULT_BACKEND=redis://localhost:6379/0
  4. 启动 Worker（新终端）：
       celery -A src.celery_app worker --loglevel=info --pool=solo
     Windows 必须用 --pool=solo（线程池），否则 fork 不可用。
  5. uvicorn serve:app 正常启动，任务自动走 Celery。

Worker 进程内执行 run_task，事件写到 outputs/jobs/{job_id}.jsonl，
SSE 端点轮询同一份日志 → 多进程架构下进度依然实时可见。
"""

import sys  # 把 src/ 加入模块搜索路径，使 Worker 进程能动态导入 pipeline 等模块
from pathlib import Path  # 计算项目根目录（定位 src/ 与 outputs/）

from celery import Celery  # Celery 分布式任务框架：任务经 Redis 投递到独立 Worker 进程

ROOT = Path(__file__).parent.parent  # 项目根目录（本文件在 src/ 下）
sys.path.insert(0, str(ROOT / "src"))  # 把 src/ 插到 sys.path 最前（Worker 内延迟导入才能找到模块）

# ═════════════════════════════════════════════════════════════════
# Celery 应用实例：配置 broker/backend 与序列化方式
# serve.py 检测 TASK_BACKEND=celery 时，任务提交从进程内线程队列切到这里
# ═════════════════════════════════════════════════════════════════
app = Celery("bp_monitor_service_agent")  # 创建 Celery 应用（名称用于任务注册与日志标识）
app.conf.update(  # 集中更新 Celery 配置项
    broker_url=__import__("os").environ.get(  # 消息 broker 地址（Redis），任务经它排队投递
        "CELERY_BROKER_URL", "redis://localhost:6379/0"),
    result_backend=__import__("os").environ.get(  # 结果后端地址（Redis），存储任务返回值
        "CELERY_RESULT_BACKEND", "redis://localhost:6379/0"),
    task_serializer="json",  # 任务参数序列化为 JSON（跨进程可读）
    result_serializer="json",  # 任务结果同样序列化为 JSON
    accept_content=["json"],  # 仅接受 JSON 内容类型（拒绝不安全反序列化）
    task_track_started=True,  # 任务开始即上报 STARTED 状态（前端可感知"已开跑"）
    worker_pool="solo",          # Windows 兼容
    task_ignore_result=False,  # 保留任务结果（供 /state 接口恢复状态用）
)


# ═════════════════════════════════════════════════════════════════
# celery_run_task：Worker 进程内执行流水线任务（pipeline.run_task 的包装）
# 关键设计：事件照常写 outputs/jobs/{job_id}.jsonl，SSE 端点轮询同一份
# 日志 → 即便执行进程与 Web 进程分离，前端进度条依然实时可见
# ═════════════════════════════════════════════════════════════════
@app.task(name="pipeline.celery_run_task", bind=True)  # 注册为 Celery 任务；bind=True 使首参 self 拿到任务实例
def celery_run_task(self, job_id: str, task: str, params: dict):
    """Worker 内执行 pipeline 任务；所有事件落盘到 job 的 JSONL 日志。"""
    from pipeline import run_task  # 延迟导入：等 sys.path 就绪后再加载，避免 Worker 启动期的导入顺序问题
    import task_queue as tq  # 延迟导入事件日志模块（EventLog 负责带序号写 JSONL）

    log = tq.EventLog(job_id)          # 单一日志实例，保证 seq 连续
    log.append("job_started", task=task, params=params)  # 记录任务启动事件（含参数快照，SSE 首条进度）
    try:
        result = run_task(task, params, emit=log.append)  # 执行流水线任务；emit 回调把每个进度事件实时落盘
        log.append("job_done", result=result)  # 记录完成事件（携带最终准确率等指标）
        return result  # 返回值同时写入 Redis result backend（供任务结果查询）
    except Exception as e:
        log.append("job_error", message=str(e)[:500])  # 异常信息截断 500 字符写入日志（前端可见失败原因）
        raise  # 继续上抛，让 Celery 把任务标记为失败（便于监控告警）
