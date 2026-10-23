"""API Key 双级鉴权单元测试（自建独立小 app，不碰业务 app）。

用例间隔离关键：auth._auth_enabled 是模块级开关，install_auth 配置过 Key 后
保持 True，会跨用例污染（后续"未配置放行"用例会被误拦截）——autouse fixture
在每个用例结束后还原 False，使用例不依赖执行顺序。
"""

import pytest  # fixture/monkeypatch
from fastapi import Depends, FastAPI, Request  # 自建最小 app + 管理守卫依赖
from fastapi.testclient import TestClient  # 内存内 HTTP 客户端

import auth  # 先导模块：还原模块级开关需要直接操作
from auth import install_auth, require_admin  # 被测入口


def _admin_guard(request: Request) -> None:
    """管理守卫依赖适配器（与 serve.py 同款）：为 require_admin 的 request
    形参补 Request 类型标注，否则 FastAPI 会把它当查询参数返回 422。"""
    require_admin(request)


@pytest.fixture(autouse=True)
def _reset_auth_flag():
    """用例间还原模块级鉴权开关（install_auth 的全局状态防跨用例污染）。"""
    yield
    auth._auth_enabled = False  # 还原"本地开发放行"形态


def _make_app() -> FastAPI:
    """最小 app：一个普通端点 + 一个管理端点（鉴权行为与业务 app 同款）。"""
    app = FastAPI()

    @app.get("/ping")
    async def ping():
        return {"ok": True}

    @app.get("/admin/secret", dependencies=[Depends(_admin_guard)])
    async def secret():
        return {"ok": True}

    return app


def _install_with_keys(monkeypatch) -> FastAPI:
    """配置双级 Key 并注册鉴权（各用例自建新 app，install 时读环境变量）。"""
    monkeypatch.setenv("BP_API_KEYS", "user-key-1")
    monkeypatch.setenv("BP_ADMIN_KEYS", "admin-key-1")
    app = _make_app()
    install_auth(app)
    return app


# ── 未认证：缺 Key / 错 Key 一律 401（不区分，防探测） ─────────────────────────

def test_missing_key_rejected(monkeypatch):
    """未携带 X-API-Key → 401。"""
    app = _install_with_keys(monkeypatch)
    assert TestClient(app).get("/ping").status_code == 401


def test_wrong_key_rejected(monkeypatch):
    """错误 Key → 401（与缺 Key 同响应，不泄露"Key 存在性"）。"""
    app = _install_with_keys(monkeypatch)
    r = TestClient(app).get("/ping", headers={"X-API-Key": "wrong-key"})
    assert r.status_code == 401


# ── 认证通过：普通 Key / 管理 Key 均 200 ──────────────────────────────────────

def test_user_key_passes(monkeypatch):
    """普通 Key → 业务端点 200。"""
    app = _install_with_keys(monkeypatch)
    r = TestClient(app).get("/ping", headers={"X-API-Key": "user-key-1"})
    assert r.status_code == 200


def test_admin_key_passes(monkeypatch):
    """管理 Key → 业务端点 200（admin 隐含普通权限）。"""
    app = _install_with_keys(monkeypatch)
    r = TestClient(app).get("/ping", headers={"X-API-Key": "admin-key-1"})
    assert r.status_code == 200


# ── 管理守卫：require_admin 对普通 Key 403、管理 Key 放行 ──────────────────────

def test_require_admin_forbids_user_key(monkeypatch):
    """普通 Key 访问管理端点 → 403（角色不足而非未认证）。"""
    app = _install_with_keys(monkeypatch)
    client = TestClient(app)
    assert client.get("/admin/secret", headers={"X-API-Key": "user-key-1"}).status_code == 403
    assert client.get("/admin/secret", headers={"X-API-Key": "admin-key-1"}).status_code == 200


def test_exempt_path_passes_without_key(monkeypatch):
    """豁免路径（/health、/）无 Key 也放行（探针与前端页面不受鉴权影响）。"""
    app = _install_with_keys(monkeypatch)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/")
    async def index():
        return {"page": "index"}

    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.get("/").status_code == 200


# ── 未配置 Key：鉴权关闭（本地开发放行） ─────────────────────────────────────

def test_disabled_when_no_keys_configured(monkeypatch):
    """BP_API_KEYS/BP_ADMIN_KEYS 均未配置 → 不注册中间件，全部端点放行。"""
    monkeypatch.delenv("BP_API_KEYS", raising=False)
    monkeypatch.delenv("BP_ADMIN_KEYS", raising=False)
    app = _make_app()
    assert install_auth(app) is None  # 未注册中间件（返回 None 是关闭形态的契约）
    client = TestClient(app)
    assert client.get("/ping").status_code == 200  # 业务端点放行
    assert client.get("/admin/secret").status_code == 200  # require_admin 未启用也放行
