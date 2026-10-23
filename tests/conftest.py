"""pytest 全局配置：把测试环境固定为零外部依赖的离线形态。

必须在导入任何 src 模块**之前**执行（conftest 先于测试模块加载）：
  - hash 嵌入：不下载模型、不调 API，结果确定性可断言
  - inprocess 队列：不需要 Redis
  - rule 评估：不需要 LLM Key（LLM Judge 层由 evaluator 单测覆盖判定逻辑）
  - 本地向量库：不需要 Milvus

如需针对 api/local/celery 后端做集成测试，请编写独立标记的用例并在
CI 的 services（redis/milvus）中运行，不要改动这里的默认值。
"""

import os

# 强制赋值（而非 setdefault）：测试确定性优先于宿主机环境，避免开发者
# 本机配置的 API Key / 模型目录让测试走到不可复现的分支
os.environ["EMBED_PROVIDER"] = "hash"
os.environ["TASK_BACKEND"] = "inprocess"
os.environ["EVAL_MODE"] = "rule"
os.environ["MILVUS_ENABLED"] = "0"
os.environ["MILVUS_URI"] = ""
os.environ["BP_EMBED_MODEL_DIR"] = ""
os.environ["EMBED_API_KEY"] = ""
os.environ["DEEPSEEK_API_KEY"] = ""
os.environ["JUDGE_API_KEY"] = ""
