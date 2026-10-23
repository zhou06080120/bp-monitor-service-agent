"""可观测性单元测试：/metrics 端点、X-Request-ID 透传、指标计数增量、
JsonFormatter、setup_logging 与 llm_call_timer 埋点。

约定（与现有测试对齐）：
  - prometheus 指标是全进程全局单例（serve.app 与测试自建 app 共用同一注册表），
    断言一律用"增量"而非绝对值
  - TestClient 不用 with：避免 lifespan 触发 _restore() 清空实验现场
  - 只打 GET /health、GET /metrics 与 404 路径——绝不触发会真实提交后台任务的接口
    （POST /job/xxx 会启动后台评估任务，严禁在单测中调用）
"""

import json  # 解析 JsonFormatter 输出的日志行
import logging  # 构造 LogRecord / 检查根 logger 的 handler
import re  # 校验 12 位十六进制 request id 形状

import pytest  # pytest.raises 验证异常原样上抛
from fastapi import FastAPI  # 限流联测自建独立 app（不受 conftest 的 RPM=0 影响）
from fastapi.testclient import TestClient  # 内存内 HTTP 客户端

import serve  # 参照 test_api.py：导入即完成限流 + 可观测中间件注册
from observability import (JsonFormatter, install_observability, llm_call_timer,
                           request_id_var, setup_logging)
from rate_limit import install_rate_limit  # 限流联测：先限流后可观测的注册顺序


client = TestClient(serve.app)  # 不用 with：跳过 lifespan 组件预热

_HEX12 = re.compile(r"^[0-9a-f]{12}$")  # 自动生成的 request id 形状（uuid4 截断 12 位）


def _counter_value(text: str, name: str, label: str | None = None) -> float:
    """从 /metrics 文本累加指标 name 的当前值。

    只认以 name+"{" 或 name+" " 开头的样本行（精确前缀匹配），
    排除 http_requests_total_created / *_bucket 这类衍生序列；
    label 可选过滤标签子串（如只统计 status="ok"）。"""
    total = 0.0  # 累加器
    for line in text.splitlines():  # 逐行扫描 Prometheus 文本协议
        if line.startswith("#"):  # HELP/TYPE 元数据行
            continue  # 跳过
        if not (line.startswith(name + "{") or line.startswith(name + " ")):  # 前缀不精确匹配
            continue  # 衍生序列（_created/_sum/_count 等）不计入
        if label is not None and label not in line:  # 标签过滤未命中
            continue  # 跳过该样本
        total += float(line.rsplit(" ", 1)[-1])  # 行尾数值累加
    return total  # 返回该指标（可选标签维度）的合计值


def _metrics() -> str:
    """拉取当前 /metrics 文本（全进程指标注册表的即时快照）。"""
    return client.get("/metrics").text  # /metrics 限流豁免，随时可抓


# ── /metrics 端点 ──────────────────────────────────────────────────────────────

def test_metrics_endpoint_exposes_prometheus_text():
    """GET /metrics：200 + Prometheus 文本协议 + 核心指标已注册。"""
    r = client.get("/metrics")
    assert r.status_code == 200  # 抓取端点可用
    assert "text/plain" in r.headers["content-type"]  # Prometheus 文本协议类型
    assert "http_requests_total" in r.text  # HTTP 请求计数器已暴露
    assert "rate_limit_rejected_total" in r.text  # 限流拒绝计数器已暴露


# ── X-Request-ID：生成 / 透传 ─────────────────────────────────────────────────

def test_health_generates_12hex_request_id():
    """客户端不带 ID 时自动生成 12 位十六进制并回写响应头。"""
    r = client.get("/health")
    assert r.status_code == 200  # 探针端点可用
    assert _HEX12.match(r.headers.get("x-request-id", ""))  # 形状：12 位小写十六进制


def test_request_id_passthrough_from_client():
    """客户端带 X-Request-ID: abc123 时响应头原样回传（全链路对账锚点）。"""
    r = client.get("/health", headers={"X-Request-ID": "abc123"})
    assert r.headers["X-Request-ID"] == "abc123"  # 透传不重写


# ── 请求计数增量 ───────────────────────────────────────────────────────────────

def test_http_requests_counter_increments_after_health():
    """打一次 /health 后 http_requests_total 计数增加（增量断言，不碰绝对值）。"""
    before = _counter_value(_metrics(), "http_requests_total")  # 打点前的合计值
    client.get("/health")  # 触发一次被统计的请求
    after = _counter_value(_metrics(), "http_requests_total")  # 打点后的合计值
    assert after - before >= 1  # 至少计入了这一次 /health


# ── 限流 429 联测（fresh app：先限流后可观测） ─────────────────────────────────

def test_rate_limit_rejected_counted_on_fresh_app(monkeypatch):
    """RPM=1 独立 app：第二次 /ping 429 且 rate_limit_rejected_total 增量 ≥1。

    同时验证两件事：
      - 中间件注册顺序（先 rate_limit 后 observability）使 429 流经可观测层被统计
      - /metrics 属限流豁免路径：被限流后仍可正常抓取
    """
    monkeypatch.setenv("RATE_LIMIT_RPM", "1")  # 阈值压到 1：第二个请求即被拒
    app = FastAPI()  # 独立 app：环境变量在 install 时读取

    @app.get("/ping")
    async def ping():
        return {"ok": True}

    install_rate_limit(app)  # 先注册限流（内层）
    install_observability(app)  # 后注册可观测（外层）：429 才能流经被统计
    c = TestClient(app)  # TestClient 恒定同一来源 IP（testclient）

    assert c.get("/ping").status_code == 200  # 窗口内首个请求放行
    before = _counter_value(c.get("/metrics").text, "rate_limit_rejected_total")  # 拒绝前计数
    assert c.get("/ping").status_code == 429  # 第二个请求超阈值被拒
    after = _counter_value(c.get("/metrics").text, "rate_limit_rejected_total")  # 拒绝后计数
    assert after - before >= 1  # 429 已计入限流拒绝指标
    assert c.get("/metrics").status_code == 200  # /metrics 豁免：被限流后仍可抓取


# ── JsonFormatter / setup_logging ──────────────────────────────────────────────

def test_json_formatter_fields_and_extra_passthrough():
    """request_id 上下文注入日志行；extra 自定义字段原样透传进 JSON。"""
    token = request_id_var.set("rid-abc")  # 模拟请求中间件注入的 request_id
    try:
        record = logging.LogRecord(name="obs-test", level=logging.INFO,  # 构造一条日志记录
                                   pathname=__file__, lineno=1,
                                   msg="hello %s", args=("world",), exc_info=None)
        setattr(record, "job_id", "job-xyz")  # 模拟 logger.info(..., extra={"job_id": ...})
        payload = json.loads(JsonFormatter().format(record))  # 格式化并反序列化
        assert payload["msg"] == "hello world"  # 消息（含 % 参数插值）
        assert payload["request_id"] == "rid-abc"  # contextvars 注入的请求 ID
        assert payload["level"] == "INFO"  # 日志级别
        assert payload["job_id"] == "job-xyz"  # extra 业务字段透传
    finally:
        request_id_var.reset(token)  # 清理上下文，防串号


def test_setup_logging_text_mode_then_restore_json():
    """setup_logging(fmt="text") 不抛异常；测试末尾恢复 JSON 默认形态。"""
    setup_logging(fmt="text")  # 本地开发人读格式
    root = logging.getLogger()  # 根 logger（setup_logging 直接操作它）
    assert not any(isinstance(h.formatter, JsonFormatter) for h in root.handlers)  # 已切文本格式
    setup_logging(fmt="json")  # 恢复默认 JSON 行式（不污染后续用例）
    assert any(isinstance(h.formatter, JsonFormatter) for h in root.handlers)  # 已恢复


# ── llm_call_timer 埋点 ────────────────────────────────────────────────────────

def test_llm_call_timer_counts_ok():
    """正常路径：with 结束后 llm_calls_total{status="ok"} 增量 ≥1。"""
    before = _counter_value(_metrics(), "llm_calls_total", 'status="ok"')  # 埋点前
    with llm_call_timer():  # 模拟一次成功的 LLM 调用
        pass
    after = _counter_value(_metrics(), "llm_calls_total", 'status="ok"')  # 埋点后
    assert after - before >= 1  # 成功计数已累计


def test_llm_call_timer_counts_error_and_reraises():
    """异常路径：llm_calls_total{status="error"} 增量 ≥1 且异常原样上抛（不吞错）。"""
    before = _counter_value(_metrics(), "llm_calls_total", 'status="error"')  # 埋点前
    with pytest.raises(RuntimeError):  # 异常必须穿透 llm_call_timer 上抛
        with llm_call_timer():  # 模拟一次失败的 LLM 调用
            raise RuntimeError("boom")  # 调用方既有的重试/降级逻辑依赖异常上抛
    after = _counter_value(_metrics(), "llm_calls_total", 'status="error"')  # 埋点后
    assert after - before >= 1  # 失败计数已累计


# ── 路由模板标签（防高基数） ───────────────────────────────────────────────────

def test_unmatched_route_uses_unmatched_label():
    """GET 不存在的路径 → /metrics 文本含 path="unmatched"（404 无路由模板）。"""
    r = client.get("/definitely-not-a-route-404")  # 只允许 GET 404 路径（无副作用）
    assert r.status_code == 404  # 确认确实未匹配路由
    assert 'path="unmatched"' in _metrics()  # 未匹配路由统一记 unmatched 标签
