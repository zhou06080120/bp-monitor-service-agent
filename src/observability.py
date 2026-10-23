"""
结构化日志 + Prometheus 指标 —— P2 可观测性基建

与姊妹项目（blood_pressure_agent / blodd_answer_piper）保持同款口径
（同步策略与 rate_limit.py 一致），提供三个能力：

  1. setup_logging()：JSON 行式结构化日志（默认；BP_LOG_FORMAT=text 切回人读格式），
     每行自动携带 request_id（请求中间件经 contextvars 注入）与 extra 业务字段，
     供 Docker / Loki 等日志采集器直接解析
  2. install_observability(app)：注册最外层 HTTP 中间件 + /metrics 抓取端点
       - 生成或透传 X-Request-ID（响应头回写，全链路日志按它串联一次请求）
       - http_requests_total{method,path,status}：按路由模板统计请求数
         （path 取 /session/{id} 这类模板，避免高基数标签爆炸；未匹配路由记 "unmatched"）
       - http_request_duration_seconds{path}：请求延迟直方图
       - rate_limit_rejected_total：429 计数（本项目 429 唯一来源是限流中间件）
       - /metrics 输出 Prometheus 文本协议（限流豁免，见 rate_limit.EXEMPT_PATHS）
  3. llm_call_timer()：LLM 调用统一埋点上下文管理器，用法：
         with llm_call_timer():
             resp = client.chat.completions.create(...)
     自动统计 llm_calls_total{status=ok|error} 与 llm_call_duration_seconds

注册顺序约定：install_observability(app) 必须在 install_rate_limit(app) 之后调用——
FastAPI 后注册的 HTTP 中间件位于外层，这样限流器返回的 429 响应
也会流经本中间件被统计进指标。
"""

import contextlib      # llm_call_timer 用 contextmanager 装饰
import contextvars     # 协程/线程安全的 request_id 上下文传播
import json            # 日志行的 JSON 序列化
import logging         # 日志框架（JsonFormatter 挂在根 logger 上）
import os              # 环境变量（BP_LOG_FORMAT / BP_LOG_LEVEL）
import time            # perf_counter 高精度计时
import uuid            # request_id 生成（uuid4 截断 12 位十六进制）

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

# request_id 上下文变量：请求中间件 set / 请求结束 reset；
# async 任务与 asyncio.to_thread 自动继承（run_in_executor 线程池不继承，需手动传递）
request_id_var = contextvars.ContextVar("request_id", default="-")

# ── Prometheus 指标（模块级单例：全进程共享，多 app 实例共用同一注册表）──────
HTTP_REQUESTS = Counter(                       # HTTP 请求计数器（按方法/路径模板/状态码）
    "http_requests_total", "HTTP 请求总数", ["method", "path", "status"])
HTTP_LATENCY = Histogram(                      # HTTP 请求延迟直方图（桶覆盖 5ms~10s）
    "http_request_duration_seconds", "HTTP 请求延迟（秒）", ["path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0))
RATE_LIMIT_REJECTED = Counter(                 # 限流拒绝计数（429 唯一来源是限流器）
    "rate_limit_rejected_total", "限流拒绝（HTTP 429）总数")
LLM_CALLS = Counter(                           # LLM 调用计数（ok / error 两种状态）
    "llm_calls_total", "LLM 调用总数", ["status"])
LLM_LATENCY = Histogram(                       # LLM 调用延迟直方图（桶覆盖 0.1s~60s）
    "llm_call_duration_seconds", "LLM 调用延迟（秒）",
    buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0))

# logging.LogRecord 的框架保留属性名：之外的属性都视作 extra={...} 业务字段透传
_RESERVED = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName", "message", "asctime",
}


class JsonFormatter(logging.Formatter):
    """把每条日志序列化成一行 JSON：ts/level/logger/msg/request_id + extra 字段。"""

    def format(self, record):
        payload = {  # 固定字段：时间/级别/模块/消息/请求 ID
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id_var.get(),   # 无请求上下文时为 "-"
        }
        for key, value in record.__dict__.items():  # extra={...} 的业务字段全部并入
            if key not in _RESERVED and key not in payload:
                payload[key] = value
        if record.exc_info:                      # 异常堆栈单独成字段（多行文本）
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)  # default 兜底不可序列化对象


def setup_logging(level: str | None = None, fmt: str | None = None) -> None:
    """初始化根日志器：默认 JSON 行式，BP_LOG_FORMAT=text 切人读格式。

    level 缺省读环境变量 BP_LOG_LEVEL（默认 INFO）。
    """
    level = level or os.getenv("BP_LOG_LEVEL", "INFO")         # 日志级别（INFO/WARNING/...）
    fmt = (fmt or os.getenv("BP_LOG_FORMAT", "json")).lower()  # 输出格式：json（默认）/ text
    handler = logging.StreamHandler()                           # 输出到标准错误流（容器友好）
    if fmt == "text":                                           # 本地开发：人读单行文本
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    else:                                                       # 生产默认：一行一条 JSON
        handler.setFormatter(JsonFormatter())
    root = logging.getLogger()                                  # 根 logger：全部子 logger 继承
    root.handlers[:] = [handler]                                # 替换全部既有 handler（含默认配置）
    root.setLevel(level.upper())                                # 统一日志级别


@contextlib.contextmanager
def llm_call_timer():
    """LLM 调用统一埋点：with 包住一次 client.chat.completions.create(...)。

    正常结束计 status=ok；抛异常计 status=error 并原样上抛（不吞错，
    交由调用方既有的重试/降级逻辑处理）。
    """
    start = time.perf_counter()                             # 调用起始时刻
    status = "ok"                                           # 先假定成功
    try:
        yield                                               # ── 包住真正的 LLM 调用 ──
    except Exception:
        status = "error"                                    # 异常：计失败
        raise                                               # 原样上抛，不吞错
    finally:
        LLM_CALLS.labels(status).inc()                      # 成败计数（ok/error）
        LLM_LATENCY.observe(time.perf_counter() - start)    # 耗时观测


def install_observability(app):
    """注册可观测中间件（最外层）与 /metrics 端点；须在 install_rate_limit 之后调用。

    中间件职责：生成/透传 X-Request-ID → 计时 → 放行 → 按路由模板记账。
    注意 SSE 流式接口的延迟只统计到响应头返回（流式体不走本计时）。
    """
    from fastapi.responses import Response  # 局部导入：与 rate_limit.py 口径一致

    @app.middleware("http")                 # 后注册者在外层：包住限流中间件
    async def _observe_request(request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]  # 透传网关 ID 或自造
        token = request_id_var.set(rid)     # 注入上下文：本请求内所有日志自动携带
        start = time.perf_counter()         # 请求起始时刻
        status = "500"                      # 兜底：call_next 抛异常时按 500 记账
        try:
            response = await call_next(request)    # 放行到内层中间件/路由
            status = str(response.status_code)     # 实际状态码（含限流 429）
            response.headers["X-Request-ID"] = rid  # 回写响应头：客户端/日志可对账
            return response
        finally:                            # 无论正常/异常都完成记账
            route = request.scope.get("route")     # 路由解析后 scope 里有模板（404 时无）
            path = getattr(route, "path", None) or "unmatched"  # 用模板防标签基数爆炸
            HTTP_REQUESTS.labels(request.method, path, status).inc()        # 请求计数
            HTTP_LATENCY.labels(path).observe(time.perf_counter() - start)  # 延迟观测
            if status == "429":             # 429 唯一来源是限流中间件
                RATE_LIMIT_REJECTED.inc()   # 单独计数便于告警配额
            request_id_var.reset(token)     # 清理上下文，防串号

    @app.get("/metrics", include_in_schema=False)  # 不进 OpenAPI 文档（运维端点）
    async def _prometheus_metrics():
        """Prometheus 抓取端点（文本协议；限流豁免由 rate_limit.EXEMPT_PATHS 保证）"""
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
