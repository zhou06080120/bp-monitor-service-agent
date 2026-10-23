"""
请求限流中间件 —— 按客户端 IP 的滑动窗口限流器

设计目标：
  - 零第三方依赖：纯标准库实现（不引入 slowapi 等新依赖）
  - 滑动窗口日志算法：每 IP 维护请求时间戳队列，无固定窗口边界突刺问题
  - 环境变量 RATE_LIMIT_RPM 控制每分钟允许的请求数（默认 60；设 0 关闭限流）
  - /health 探针端点豁免（监控探针不应被限流）
  - 内存自愈：跟踪 IP 超过上限时清理已过期空队列，防慢速扫描撑爆内存

注意：生产部署在反向代理（Nginx）后所有请求直连 IP 相同，需在代理层
配置真实 IP 透传（X-Forwarded-For），本中间件按直连 IP 计数。
"""

import os   # 读 RATE_LIMIT_RPM 配置
import math  # retry_after 向上取整
import time  # 单调时钟计时
import threading  # 保护并发计数的锁
from collections import defaultdict, deque  # ip → 时间戳队列

DEFAULT_RPM = 60          # 默认限流阈值：每分钟 60 次请求（环境变量可覆盖）
EXEMPT_PATHS = {"/health"}  # 限流豁免路径：健康检查等探针端点
MAX_TRACKED_IPS = 10000   # IP 跟踪表规模上限（超过即触发过期清理）


class RateLimiter:
    """按 key（客户端 IP）的滑动窗口限流器（线程安全）。"""

    def __init__(self, rpm: int, window_s: float = 60.0):
        self.rpm = max(0, int(rpm))   # 窗口内允许的最大请求数（0 = 关闭限流）
        self.window_s = window_s      # 滑动窗口长度（秒）
        self._hits = defaultdict(deque)  # key → 该窗口内的请求时间戳队列
        self._lock = threading.Lock()    # 保护 _hits 的并发读写

    def allow(self, key: str, now: float | None = None) -> bool:
        """判断 key 本次请求是否放行（放行则记录本次时间戳）。"""
        if self.rpm <= 0:             # 限流关闭：一律放行
            return True
        now = time.monotonic() if now is None else now  # 单调时钟（不受系统时间回拨影响）
        with self._lock:              # 全程持锁：淘汰、判满、记录需原子完成
            q = self._hits[key]       # 该 key 的时间戳队列（defaultdict 自动建）
            cutoff = now - self.window_s  # 窗口左边界
            while q and q[0] <= cutoff:   # 淘汰已滑出窗口的旧时间戳
                q.popleft()
            if len(q) >= self.rpm:    # 窗口内请求数已满
                return False          # 拒绝（不记录被拒请求）
            q.append(now)             # 记录本次放行
            if len(self._hits) > MAX_TRACKED_IPS:  # IP 表过大：清理过期队列自愈
                self._evict_stale(now)
            return True

    def retry_after(self, key: str, now: float | None = None) -> int:
        """该 key 距离下一个可放行时刻还需等待的秒数（向上取整，至少 1 秒）。"""
        now = time.monotonic() if now is None else now
        with self._lock:
            q = self._hits.get(key)   # 该 key 的队列（可能不存在）
            if not q:                 # 无记录（如限流已关闭）
                return 1
            wait = q[0] + self.window_s - now  # 最早一条滑出窗口的时刻即可放行
        return max(1, math.ceil(wait))  # 秒级向上取整，至少 1

    def _evict_stale(self, now: float) -> None:
        """清理已完全过期的 IP 队列（调用方需持锁）。"""
        cutoff = now - self.window_s  # 早于此的时间戳已全部过期
        for k in [k for k, v in self._hits.items() if not v or v[-1] <= cutoff]:
            del self._hits[k]         # 删除空队列或全部时间戳已过期的队列


def install_rate_limit(app, env_var: str = "RATE_LIMIT_RPM", default_rpm: int = DEFAULT_RPM):
    """在 FastAPI 应用上注册限流中间件，返回 limiter 实例（测试可访问）。

    rpm <= 0（关闭限流）时不注册中间件，零开销。
    """
    from fastapi import Request  # 局部导入：纯单测导入本模块不连带拉起 FastAPI
    from fastapi.responses import JSONResponse  # 429 响应体

    limiter = RateLimiter(int(os.getenv(env_var, str(default_rpm))))  # 环境变量读阈值

    if limiter.rpm <= 0:             # 显式关闭：不注册中间件
        return limiter

    @app.middleware("http")          # HTTP 中间件：请求进入路由前拦截
    async def _rate_limit_middleware(request: Request, call_next):
        if request.url.path in EXEMPT_PATHS:  # 探针端点豁免
            return await call_next(request)
        client_ip = request.client.host if request.client else "unknown"  # 直连 IP
        if not limiter.allow(client_ip):      # 超阈值：拒绝并告知等待时间
            return JSONResponse(
                status_code=429,              # Too Many Requests
                content={"detail": "请求过于频繁，请稍后再试"},
                headers={"Retry-After": str(limiter.retry_after(client_ip))},
            )
        return await call_next(request)       # 放行进入下一环节

    return limiter
