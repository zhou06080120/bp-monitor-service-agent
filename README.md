# BP Monitor Service Agent

[![CI](https://github.com/zhou06080120/bp-monitor-service-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/zhou06080120/bp-monitor-service-agent/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED?logo=docker&logoColor=white)](https://hub.docker.com/)

血压计医疗器械客服 Agent 服务：**LLM-as-a-Judge 双模态评估 + Skill 知识自进化 + 混合向量检索 + 异步任务架构**，面向"新类目分钟级上线、人工客服分流 65%+"的医疗器械客服场景。

## 核心能力

- **知识自进化闭环**：badcase 驱动 Skill 生成/修补（patch），版本历史 + 快照支持按动作类型精确回滚，种子 Skill 永不丢失
- **双模态联合校验**：LLM-as-a-Judge 语义裁决 + 规则评估互为补充，替代死板的关键词匹配
- **混合向量检索**：Dense（Milvus ANN / 本地余弦）+ Sparse（BM25）+ RRF 融合，Skill 分块按需动态挂载，控制上下文成本
- **异步任务架构**：评估/进化闭环经 Celery 剥离至后台 Worker，前端 SSE 无阻塞流式进度
- **零依赖降级链**：无 API Key / 无 Milvus / 无 Redis 时自动落到 hash 嵌入 + 本地向量库 + 进程内队列，任何环境可启动

## 快速开始

```bash
# 1. 安装（Python 3.10+）
pip install -e ".[dev]"

# 2. 启动服务（零配置即离线形态）
uvicorn serve:app --host 0.0.0.0 --port 8000

# 3. 运行测试（79 个用例，全程离线）
pytest --cov
```

**Docker 方式：**

```bash
docker compose up -d                          # API 服务（默认离线形态）
docker compose --profile celery up -d         # + Redis + Celery Worker
docker compose --profile milvus up -d         # + Milvus 向量库
```

生产配置（LLM / Embedding / Milvus / Celery）参考 [.env.example](.env.example)。

## 生产加固

- **请求限流**：`RATE_LIMIT_RPM` 控制每分钟每 IP 请求上限（默认 60，`0` 关闭；`/health` 探针豁免），超限返回 429 + `Retry-After`
- **优雅关停**：应用退出时向后台 Worker 投递停机哨兵，在跑任务执行完毕后线程才退出（超时不强杀）
- **LLM 韧性**：全部 LLM 调用自带超时（`LLM_TIMEOUT`）与重试（`LLM_MAX_RETRIES`），上游抖动不拖垮请求

## 医疗合规

面向医疗器械客服场景的合规底线，统一实现在 `src/compliance.py`（纯标准库零依赖，姊妹项目同款口径）：

- **免责声明策略**：全站统一医疗免责声明常量 `MEDICAL_DISCLAIMER`——"本回答仅供健康教育参考，不能替代执业医师诊疗意见；如有不适或异常读数，请及时就医"；前端页脚（`index.html`）固定展示同一文案并附带"以上内容由 AI 生成"标识
- **输入校验**：`validate_query()` 对用户输入做清洗——空/纯空白拒绝、控制字符剥离（防日志注入）、500 字上限（超长多为滥用）；`check_injection()` 拦截高置信 Prompt 注入（指令覆盖 / 角色劫持 / 系统探测），只收录高置信攻击语句，宁可漏报不误伤正常医疗咨询
- **密钥零硬编码**：所有 API Key（`DEEPSEEK_API_KEY` / `JUDGE_API_KEY` / `EMBED_API_KEY`）仅经环境变量注入，模板见 [.env.example](.env.example)（密钥值只留占位符），`.env` 已被 .gitignore 忽略；无 Key 时自动降级为离线形态而非报错

本项目 API 均为路径参数驱动（无用户自由文本输入面），输入校验与注入拦截能力由 `tests/test_compliance.py` 单测锁定。

## 可观测性

结构化日志 + Prometheus 指标（`src/observability.py`），Docker / Loki / Grafana 开箱可采：

**指标表**（`GET /metrics`，Prometheus 文本协议；`/metrics` 与 `/health` 限流豁免）：

| 指标名 | 类型 | 标签 | 含义 |
| --- | --- | --- | --- |
| `http_requests_total` | Counter | method / path / status | HTTP 请求总数（path 取路由模板如 `/job/{job_id}` 防标签基数爆炸，未匹配路由记 `unmatched`） |
| `http_request_duration_seconds` | Histogram | path | HTTP 请求延迟（桶覆盖 5ms~10s） |
| `rate_limit_rejected_total` | Counter | — | 限流拒绝（HTTP 429）总数，可直接接告警 |
| `llm_calls_total` | Counter | status=ok/error | LLM 调用成败计数（Agent / Judge / Reviewer 全部经 `llm_call_timer()` 统一埋点） |
| `llm_call_duration_seconds` | Histogram | — | LLM 调用耗时（桶覆盖 0.1s~60s） |
| `python_gc_*` / `python_info` / `process_*` | — | — | prometheus_client 默认注入的解释器 / 进程运行时指标 |

**X-Request-ID 机制**：最外层中间件为每个请求生成 12 位十六进制 ID（或透传客户端带来的），回写响应头，并经 contextvars 注入本请求内所有日志行——一次请求的访问日志、后台任务日志、异常堆栈可用同一个 ID 串联对账。注意中间件注册顺序：可观测层在限流层**之后**注册（FastAPI 后注册者在外层），限流器返回的 429 才会被统计进指标。

**JSON 日志**（默认一行一条，真实样例）：

```json
{"ts": "2026-10-23T15:31:19", "level": "INFO", "logger": "serve", "msg": "[serve] 组件就绪: eval_mode=rule, embedder=hash, vector_backend=local, task_backend=inprocess", "request_id": "a1b2c3d4e5f6", "eval_mode": "rule", "embedder": "hash", "vector_backend": "local", "task_backend": "inprocess"}
```

`logger.info(..., extra={"job_id": ...})` 的业务字段会原样并入 JSON 行。本地开发可切人读格式与级别：

```bash
BP_LOG_FORMAT=text BP_LOG_LEVEL=DEBUG uvicorn serve:app   # text=单行文本（默认 json），级别默认 INFO
curl localhost:8000/metrics                               # 抓取指标（限流豁免）
curl -H "X-Request-ID: abc123" localhost:8000/health      # 响应头原样回传该 ID
```

## 评估门禁

lint/test 只能证明"代码没坏"，无法证明"评估器还判得对"。离线评估门禁在 Docker 构建前加一道语义闸门，防止改动评估器规则 / Skill 结构导致判分能力静默回退：

- **`evals/gate.py`**：三类检查——① 结构校验（每个技能目录 `SKILL.md` 存在且非空；`EventLog` 临时目录写 3 条事件读回，seq 严格递增 / ts-type 齐全 / JSONL 可解析）② 判别力校验（`golden_pairs.json` 12 对金标逐对离线打分）③ `/metrics` 冒烟。任一项 FAIL 退出码 1
- **`evals/golden_pairs.json`**：6 对好答案（含关键事实，应通过）+ 6 对坏答案（推脱 / 无关 / 错误事实，应拦截），覆盖全部 6 个类目
- **`evals/thresholds.json`**：判别力阈值（实测值留安全余量，任一实测低于阈值即门禁失败）
- **CI**：test 与 docker 之间新增 `eval-gate` job，docker 构建 `needs: [lint, test, eval-gate]`——门禁失败阻断镜像构建

实测数字（2026-10 本地离线运行，`EVAL_MODE=rule`，零 LLM 成本）：

| 指标 | 实测 | 阈值 |
| --- | --- | --- |
| good_pass_rate（好答案被判通过） | 6/6 = 1.00 | ≥ 0.9 |
| bad_block_rate（坏答案被判拦截） | 6/6 = 1.00 | ≥ 0.9 |

本地运行（无 API Key、无外部服务、不改动 outputs/ 现场）：

```bash
python evals/gate.py   # 全部 PASS 退出码 0；任一 FAIL 退出码 1
```

## CI / CD

每次 push 与 PR 自动执行三段质检（见 [ci.yml](.github/workflows/ci.yml)）：

```
Lint (ruff) ──▶ Test (py3.10 + py3.12, 79 用例, 覆盖率门槛 30%) ──▶ Docker build
```

## 项目结构

```
serve.py                 # FastAPI 入口：任务提交 + SSE 推流 + 阶段门禁
src/
  ├── agent.py           # 客服 Agent（Skill 动态挂载 + 生成回答）
  ├── llm_judge.py       # LLM-as-a-Judge 语义裁决
  ├── skill_manager.py   # Skill CRUD / 版本历史 / 精确回滚
  ├── skill_store.py     # Skill 分块 + 向量化 + 按需挂载
  ├── vector_store.py    # Local / Milvus 混合检索（Dense+BM25+RRF）
  ├── evaluator.py       # 评估器（规则 / 语义 / 混合模式）
  ├── task_queue.py      # 进程内 / Celery 可插拔任务队列
  └── ...
tests/                   # 离线单元测试（含踩坑场景回归锁定）
```
