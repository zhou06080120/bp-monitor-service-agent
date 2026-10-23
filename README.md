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

# 3. 运行测试（49 个用例，全程离线）
pytest --cov
```

**Docker 方式：**

```bash
docker compose up -d                          # API 服务（默认离线形态）
docker compose --profile celery up -d         # + Redis + Celery Worker
docker compose --profile milvus up -d         # + Milvus 向量库
```

生产配置（LLM / Embedding / Milvus / Celery）参考 [.env.example](.env.example)。

## CI / CD

每次 push 与 PR 自动执行三段质检（见 [ci.yml](.github/workflows/ci.yml)）：

```
Lint (ruff) ──▶ Test (py3.10 + py3.12, 49 用例, 覆盖率门槛 30%) ──▶ Docker build
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
