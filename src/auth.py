"""
API Key 双级鉴权中间件 —— P4 平台安全基建

与姊妹项目（blood_pressure_agent / blodd_answer_piper）保持同款口径
（同步策略与 rate_limit.py / observability.py / compliance.py 一致）。

设计要点：
  - 双级 Key：普通 Key（业务端点）与管理 Key（管理端点，隐含普通权限）
  - 环境变量：BP_API_KEYS / BP_ADMIN_KEYS（逗号分隔，各自独立集合）
  - 未配置任何 Key → 不注册中间件（本地开发/测试零影响，仅记一条 warning）
  - 豁免路径：/health、/metrics、/（前端页面）与 OPTIONS 预检请求
  - 认证通过后写入 request.state.api_role（user|admin）与
    request.state.actor_id（Key 的 SHA-256 前 8 位指纹，供审计脱敏留痕）

注册顺序约定（关键）：install_auth() 必须在 install_rate_limit() 之前调用
（洋葱模型内层），使请求顺序为 observability(最外) → rate_limit → auth → 路由：
限流先于鉴权执行可防 Key 暴力枚举，401/403 也都会被指标统计与结构化日志捕获。
"""

import hashlib  # Key 指纹脱敏（不落明文）
import logging  # 鉴权关闭时的一条 warning
import os       # 读 BP_API_KEYS / BP_ADMIN_KEYS 配置

logger = logging.getLogger(__name__)  # 本模块日志器

DEFAULT_EXEMPT_PATHS = {"/", "/health", "/metrics"}  # 豁免路径：探针 + 前端首页

_auth_enabled = False  # 鉴权是否已启用（require_admin 依赖据此判定放行/拦截）


def _parse_keys(raw: str) -> set[str]:
    """解析逗号分隔的 Key 串（去空白、去空项）。"""
    return {k.strip() for k in raw.split(",") if k.strip()}


def install_auth(app, env_keys: str = "BP_API_KEYS", env_admin_keys: str = "BP_ADMIN_KEYS",
                 exempt: "set[str] | None" = None):
    """在 FastAPI 应用上注册 API Key 鉴权中间件。

    未配置任何 Key 时不注册中间件（本地开发模式，向后兼容），返回 None；
    注册成功返回 key→role 映射表（测试可访问）。
    """
    global _auth_enabled  # 模块级开关（require_admin 依赖读取）

    from fastapi import Request   # 局部导入：纯单测导入本模块不连带拉起 FastAPI
    from fastapi.responses import JSONResponse  # 401 响应体

    user_keys = _parse_keys(os.getenv(env_keys, ""))    # 普通 Key 集合
    admin_keys = _parse_keys(os.getenv(env_admin_keys, ""))  # 管理 Key 集合

    if not user_keys and not admin_keys:  # 一个 Key 都没配
        logger.warning("未配置 %s/%s，API 鉴权关闭（本地开发模式；生产环境务必配置）",
                       env_keys, env_admin_keys)
        return None

    exempt_paths = DEFAULT_EXEMPT_PATHS | set(exempt or ())  # 豁免路径合集
    key_role = {}                       # Key → 角色映射（admin 覆盖 user，同 Key 不共存两级）
    for k in user_keys:                 # 先登普通级
        key_role[k] = "user"
    for k in admin_keys:                # 再登管理级（同 Key 出现在两个集合时以 admin 为准）
        key_role[k] = "admin"
    _auth_enabled = True                # 打开 require_admin 的拦截开关

    @app.middleware("http")             # HTTP 中间件：请求进入路由前校验 Key
    async def _auth_middleware(request: Request, call_next):
        if request.method == "OPTIONS" or request.url.path in exempt_paths:
            return await call_next(request)   # 预检/豁免路径直接放行
        key = request.headers.get("x-api-key", "")  # 从请求头取 Key
        role = key_role.get(key)         # 查角色（未知 Key 得 None）
        if role is None:                 # 缺 Key / 错 Key：统一 401（不区分两种情况，防探测）
            return JSONResponse(status_code=401, content={"detail": "缺少或无效的 API Key"})
        request.state.api_role = role    # 角色挂到请求状态（路由层可读）
        request.state.actor_id = hashlib.sha256(key.encode()).hexdigest()[:8]  # 脱敏指纹（审计用）
        return await call_next(request)  # 放行进入下一环节

    return key_role


def require_admin(request) -> None:
    """管理端点守卫（FastAPI 依赖用法：dependencies=[Depends(require_admin)]）。

    - 鉴权未启用（本地开发模式）：直接放行，保持向后兼容
    - 鉴权已启用：角色非 admin 抛 403（普通用户/未认证均拦截）
    """
    from fastapi import HTTPException  # 局部导入：与 install_auth 同策略

    if not _auth_enabled:               # 鉴权关闭：本地开发放行
        return
    if getattr(request.state, "api_role", None) != "admin":  # 非 admin（含未认证直访）
        raise HTTPException(status_code=403, detail="该操作需要管理员 API Key")
