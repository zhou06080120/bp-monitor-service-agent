"""请求限流单元测试：滑动窗口语义 + 关闭限流 + 过期清理 + 中间件集成。

时间全部经 now 参数注入（假想单调时钟值），绝不真实 sleep：
  - rpm=3：三连放行第四次拒绝
  - 窗口滑动：旧时间戳滑出窗口后恢复放行（无固定窗口边界突刺）
  - rpm=0：限流关闭，一律放行
  - retry_after：距下一个可放行时刻的等待秒数（向上取整）
  - _evict_stale：IP 跟踪表超限时清理过期队列（内存自愈）
  - 中间件集成：429 + Retry-After 响应头；/health 探针豁免
"""

from collections import deque  # 手工构造 _hits 内部队列

from fastapi import FastAPI  # 限流专项测试自建独立 app（不受 conftest 的 RPM=0 影响）
from fastapi.testclient import TestClient  # 内存内 HTTP 客户端

import rate_limit  # 先导模块：monkeypatch MAX_TRACKED_IPS 需要改模块级常量
from rate_limit import RateLimiter, install_rate_limit


# ── 滑动窗口语义 ───────────────────────────────────────────────────────────────

def test_allows_up_to_rpm_then_rejects():
    """rpm=3：前 3 次放行，第 4 次拒绝。"""
    limiter = RateLimiter(rpm=3)
    assert limiter.allow("1.2.3.4", now=100.0) is True
    assert limiter.allow("1.2.3.4", now=100.5) is True
    assert limiter.allow("1.2.3.4", now=101.0) is True
    assert limiter.allow("1.2.3.4", now=101.5) is False  # 窗口内已满 3 条


def test_sliding_window_recovers_after_expiry():
    """窗口滑动：rpm=2，now=0 两次放行第三次拒；now=61 最早记录滑出窗口后再放行。"""
    limiter = RateLimiter(rpm=2)
    assert limiter.allow("ip-a", now=0.0) is True
    assert limiter.allow("ip-a", now=0.0) is True
    assert limiter.allow("ip-a", now=0.0) is False  # 窗口内已有 2 条
    assert limiter.allow("ip-a", now=61.0) is True  # cutoff=1，两条 now=0 均已滑出


def test_rpm_zero_disables_limiting():
    """rpm=0：限流关闭，连续 100 次一律放行。"""
    limiter = RateLimiter(rpm=0)
    assert all(limiter.allow("ip-b", now=float(i)) for i in range(100))


def test_retry_after_returns_window_remaining():
    """retry_after：rpm=1，now=100 放行一次后，还需等 60s 窗口滑空。"""
    limiter = RateLimiter(rpm=1)
    assert limiter.allow("ip-c", now=100.0) is True
    assert limiter.retry_after("ip-c", now=100.0) == 60  # 恰好整窗，ceil 后仍 60


# ── 内存自愈：_evict_stale ─────────────────────────────────────────────────────

def test_evict_stale_removes_expired_and_empty_queues():
    """手工构造过期/空/存活三类队列：仅完全过期的被清理。"""
    limiter = RateLimiter(rpm=10)
    limiter._hits["stale"] = deque([1.0])            # 最后一条远早于 cutoff → 过期
    limiter._hits["empty"] = deque()                 # 空队列 → 清理
    limiter._hits["alive"] = deque([500.0, 501.0])   # 仍在窗口内 → 保留

    limiter._evict_stale(now=560.0)  # cutoff = 560 - 60 = 500

    assert "stale" not in limiter._hits  # 过期队列被清
    assert "empty" not in limiter._hits  # 空队列被清
    assert "alive" in limiter._hits      # 窗口内的保留


def test_allow_triggers_eviction_on_table_overflow(monkeypatch):
    """IP 跟踪表超 MAX_TRACKED_IPS 时，allow 自动清理过期队列。"""
    monkeypatch.setattr(rate_limit, "MAX_TRACKED_IPS", 2)  # 压低上限便于触发
    limiter = RateLimiter(rpm=10)
    limiter.allow("ip-old", now=0.0)    # 早请求（随后过期）
    limiter.allow("ip-new", now=999.0)  # 新请求

    limiter.allow("ip-third", now=999.0)  # 第 3 个 key：表超限 → 触发过期清理

    assert "ip-old" not in limiter._hits  # 过期队列被自愈清除
    assert "ip-new" in limiter._hits      # 活跃队列不受影响
    assert "ip-third" in limiter._hits    # 刚记录的本次放行保留


# ── 中间件集成（自建 app，与 conftest 的 RPM=0 无关） ──────────────────────────

def test_middleware_returns_429_with_retry_after(monkeypatch):
    """RPM=3：/ping 前 3 次 200，第 4 次 429 且带 Retry-After；/health 豁免不限。"""
    monkeypatch.setenv("RATE_LIMIT_RPM", "3")
    app = FastAPI()  # 独立 app：环境变量在 install 时读取

    @app.get("/ping")
    async def ping():
        return {"ok": True}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    install_rate_limit(app)  # 此时读到 RPM=3，注册中间件
    client = TestClient(app)  # TestClient 恒定同一来源 IP（testclient）

    for _ in range(3):  # 阈值内：全部放行
        assert client.get("/ping").status_code == 200
    r = client.get("/ping")  # 第 4 次：超阈值
    assert r.status_code == 429
    assert "Retry-After" in r.headers  # 告知客户端等待秒数
    assert int(r.headers["Retry-After"]) >= 1

    for _ in range(10):  # 探针豁免：/health 打多次仍全 200（监控探针不受限流影响）
        assert client.get("/health").status_code == 200


def test_middleware_not_installed_when_disabled(monkeypatch):
    """RPM=0：install 返回 limiter 但不注册中间件（零开销关闭）。"""
    monkeypatch.setenv("RATE_LIMIT_RPM", "0")
    app = FastAPI()

    @app.get("/ping")
    async def ping():
        return {"ok": True}

    limiter = install_rate_limit(app)
    client = TestClient(app)

    assert limiter.rpm == 0  # 关闭形态
    for _ in range(20):  # 随便打多少次都放行
        assert client.get("/ping").status_code == 200
