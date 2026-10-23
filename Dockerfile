# ═════════════════════════════════════════════════════════════════
# 多阶段构建：builder 装依赖 → runtime 只带运行时产物
#   镜像默认零外部依赖形态（hash 嵌入 / 进程内队列 / 本地向量库），
#   生产增强（Milvus / Celery / 语义模型）经环境变量 + compose profile 启用。
#
# 构建：  docker build -t bp-monitor-service-agent .
# 运行：  docker run -p 8000:8000 bp-monitor-service-agent
# ═════════════════════════════════════════════════════════════════

# ── 阶段 1：安装依赖（依赖层缓存：仅 pyproject 变化时重建） ───────────────────
FROM python:3.12-slim AS builder

WORKDIR /build
COPY pyproject.toml ./
# 虚包安装：只落地 dependencies 声明的第三方包，不打包应用代码
RUN pip install --no-cache-dir --target=/deps .

# ── 阶段 2：运行时镜像（不含 pip 缓存与构建工具，体积最小化） ─────────────────
FROM python:3.12-slim

# 安全基线：非 root 运行
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

# 依赖（来自 builder 的 site-packages）
COPY --from=builder /deps /deps

# 应用代码（serve.py 通过 sys.path 引用 src/，保持原始布局）
COPY serve.py index.html ./
COPY src ./src
COPY skills ./skills
COPY data ./data

# outputs 由 compose 挂卷持久化（Skill 快照/版本历史/实验状态）
RUN mkdir -p outputs && chown -R appuser:appuser /app

USER appuser

# 默认离线形态：任何环境都能起（企业级"可用性优先"降级策略）
ENV PYTHONPATH=/deps \
    PYTHONUNBUFFERED=1 \
    EMBED_PROVIDER=hash \
    TASK_BACKEND=inprocess \
    MILVUS_ENABLED=0

EXPOSE 8000

# 轻量健康检查：/job/active 不触发组件预热，毫秒级返回
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/job/active', timeout=3)"

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
