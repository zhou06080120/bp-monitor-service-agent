"""
向量存储层：Dense 向量检索 + Sparse(BM25) 词汇检索 → RRF 混合融合。

双后端（同一接口，按环境变量自动选择）：
  1. LocalVectorStore（默认）：进程内精确检索。
     数百个 Skill 分块规模下毫秒级，零依赖、离线可用，适合 Demo/开发。
  2. MilvusVectorStore（可选）：Dense 走 Milvus ANN 索引（IVF_FLAT/COSINE），
     支持几百个细分场景、几十万分块的规模化检索。
     启用：MILVUS_URI=tcp://localhost:19530（或 MILVUS_ENABLED=1 连默认地址）+ pip install pymilvus

Sparse（BM25）为什么默认留在应用侧：
  BM25 需要按语言分词（中文 char bigram），应用侧实现可保证与数据集
  分词一致、可复现；分块数量再大也只是 O(N) 打分，用 Top-K 堆裁剪即可。
  生产规模更大时可切换 Milvus 内置 BM25 analyzer（见 MilvusVectorStore.search_hybrid）。

混合检索 = RRF（Reciprocal Rank Fusion）：
  score(d) = Σ_{list∈{dense,sparse}} 1/(60 + rank_list(d))
  —— 无需调权重，对两个通路天然公平。
"""

import os  # 读取 Milvus 开关/地址/维度等环境变量
import math  # BM25 的 idf 对数计算
import heapq  # Top-K 堆选（避免全量排序）
from collections import Counter  # 词频/文档频统计


# ═════════════════════════════════════════════════════════════════
# _try_import：可选依赖探测——pymilvus 装了才用 Milvus，没装静默回退本地
# ═════════════════════════════════════════════════════════════════
def _try_import(module: str):
    try:
        return __import__(module)  # 动态导入指定模块
    except ImportError:  # 模块未安装
        return None  # 返回 None（调用方据此降级）


# ── BM25（应用侧稀疏检索） ────────────────────────────────────────────────────

# ═════════════════════════════════════════════════════════════════
# BM25Index —— 纯 Python BM25 稀疏检索（k1=1.5, b=0.75 经验参数）
# 分词复用 embeddings.tokenize（中文 unigram+bigram），保证检索侧
# 与建库侧分词一致、离线可复现；命中精确词汇（型号/数字）时强于 Dense
# ═════════════════════════════════════════════════════════════════
class BM25Index:
    """纯 Python BM25（k1=1.5, b=0.75），中文 char unigram+bigram 分词。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b  # BM25 两个可调参数：词频饱和 k1、文档长度归一 b
        self.docs: list[list[str]] = []  # 每篇文档的 token 列表
        self.doc_len: list[int] = []  # 每篇文档长度（token 数）
        self.df: Counter = Counter()  # 文档频：token 出现在几篇文档
        self.avgdl: float = 1.0  # 平均文档长度（打分时归一用）

    def _tokenize(self, text: str):
        from embeddings import tokenize  # 复用统一分词器（延迟导入避免循环依赖）
        return tokenize(text)  # 返回 unigram+bigram token 序列

    # ═══════════════════════════════════════════════════════════
    # build：建库——分词、算文档频与平均长度（BM25 三要素）
    # ═══════════════════════════════════════════════════════════
    def build(self, texts: list[str]):
        self.docs = [self._tokenize(t) for t in texts]  # 全量文档分词
        self.doc_len = [len(d) for d in self.docs]  # 记录各文档长度
        self.df = Counter()  # 重置文档频统计
        for d in self.docs:  # 逐篇统计
            self.df.update(set(d))  # 每篇文档的 token 去重后累计 df（一篇最多贡献 1）
        self.avgdl = sum(self.doc_len) / len(self.docs) if self.docs else 1.0  # 平均文档长度（空库防除零）

    # ═══════════════════════════════════════════════════════════
    # score_all：对库里每篇文档打 BM25 分，返回与文档数等长的分数列表
    # O(查询词数 × 文档数)，规模可控；skill_store.retrieve 拿 Top-K 用
    # ═══════════════════════════════════════════════════════════
    def score_all(self, query: str) -> list[float]:
        q = self._tokenize(query)  # 查询分词
        if not q or not self.docs:  # 空查询或空库
            return [0.0] * len(self.docs)  # 全零分（长度与文档数一致，保持对齐）
        scores = []  # 各文档得分收集器
        n = len(self.docs)  # 库内文档总数
        for i, d in enumerate(self.docs):  # 逐文档打分
            tf = Counter(d)  # 该文档的词频表
            dl = self.doc_len[i] or 1  # 文档长度（0 时取 1 防除零）
            s = 0.0  # 该文档累计得分
            for t in set(q):  # 查询 token 去重遍历
                if not tf[t]:  # 文档不含该 token
                    continue  # 跳过（贡献 0 分）
                idf = math.log(1 + (n - self.df[t] + 0.5) / (self.df[t] + 0.5))  # BM25 idf：罕见词权重高，恒为正
                s += idf * (tf[t] * (self.k1 + 1)) / (
                    tf[t] + self.k1 * (1 - self.b + self.b * dl / self.avgdl))  # BM25 主式：tf 饱和 + 文档长度归一
            scores.append(s)  # 收集该文档得分
        return scores  # 返回全量文档得分列表


# ── 本地向量库 ────────────────────────────────────────────────────────────────

def _cosine(a, b):
    return sum(x * y for x, y in zip(a, b))   # 已归一化，点积即余弦


# ═════════════════════════════════════════════════════════════════
# LocalVectorStore —— 默认后端：进程内精确检索（零依赖）
# Dense 暴力余弦 + 应用侧 BM25，各取 Top-K 后 RRF 融合；
# 数百块规模毫秒级，SkillStore 的默认挂载检索器
# ═════════════════════════════════════════════════════════════════
class LocalVectorStore:
    backend = "local"  # 后端标识（日志与 last_mount 展示）

    def __init__(self, dim: int):
        self.dim = dim  # 向量维度（须与嵌入器一致）
        self.ids: list[str] = []  # 分块 ID 列表（按下标对齐）
        self.texts: list[str] = []  # 分块文本列表
        self.metas: list[dict] = []  # 分块元数据列表（skill/section 溯源）
        self.vecs: list[list[float]] = []  # 分块向量列表
        self.bm25 = BM25Index()  # 稀疏检索索引

    def index(self, items: list[dict]) -> int:
        """重建索引。items: [{id, text, meta?}] 返回分块数。"""
        self.ids = [it["id"] for it in items]  # 抽取 ID 列
        self.texts = [it["text"] for it in items]  # 抽取文本列
        self.metas = [it.get("meta", {}) for it in items]  # 抽取元数据列（缺省空字典）
        self.vecs = [it["vec"] for it in items]  # 抽取向量列
        self.bm25.build(self.texts)  # 用全量文本重建 BM25 索引
        return len(items)  # 返回索引分块数

    # ═══════════════════════════════════════════════════════════
    # search_dense：暴力余弦检索——查询向量与全库向量逐一算分，
    # heapq.nlargest 取 Top-K（O(N log K)，免全量排序）
    # ═══════════════════════════════════════════════════════════
    def search_dense(self, qvec: list[float], k: int) -> list[dict]:
        if not self.vecs:  # 空库
            return []  # 无结果
        scored = [(_cosine(qvec, v), i) for i, v in enumerate(self.vecs)]  # 全库打分：(得分, 文档下标)
        top = heapq.nlargest(min(k, len(scored)), scored, key=lambda x: x[0])  # 堆选 Top-K（k 超库容时取全部）
        return [self._hit(i, score=s, rank=r + 1)  # 转成标准命中结构（rank 从 1 起）
                for r, (s, i) in enumerate(top)]  # enumerate 同时拿到名次与 (分, 下标)

    # ═══════════════════════════════════════════════════════════
    # search_hybrid：混合检索——Dense 与 BM25 各自 Top-K，
    # RRF 融合（1/(60+rank) 求和），无需调权；命中附双路名次便于诊断
    # ═══════════════════════════════════════════════════════════
    def search_hybrid(self, query: str, qvec: list[float], k: int = 8) -> list[dict]:
        """Dense + Sparse 两路召回各自 Top-K，RRF 融合取前 k。"""
        dense = self.search_dense(qvec, k)  # Dense 路：语义召回（dict 列表，含 id）
        sparse_scores = self.bm25.score_all(query)  # Sparse 路：BM25 全库打分
        sparse = [(s, i) for i, s in enumerate(sparse_scores) if s > 0]  # 只留正分候选（0 分无意义）
        sparse = heapq.nlargest(min(k, len(sparse)), sparse, key=lambda x: x[0])  # Sparse 路 Top-K

        rrf: dict[int, float] = {}  # RRF 融合分：文档下标 → 累积分
        id_to_i = {v: i for i, v in enumerate(self.ids)}  # ID → 下标映射（dense 命中只有 id）
        for r, h in enumerate(dense):          # dense 命中是 dict（含 id），需映射回下标
            i = id_to_i.get(h["id"])  # 映射回库内下标
            if i is not None:  # 映射成功
                rrf[i] = rrf.get(i, 0.0) + 1.0 / (60 + r + 1)  # RRF 贡献：1/(60+名次)，名次从 1 起
        for r, (_, i) in enumerate(sparse):  # sparse 命中直接是 (分, 下标)
            rrf[i] = rrf.get(i, 0.0) + 1.0 / (60 + r + 1)  # 同式累加（两路同文名次贡献相加）

        top = heapq.nlargest(min(k, len(rrf)), rrf.items(), key=lambda x: x[1])  # 按融合分取 Top-K
        dense_rank = {h["id"]: r for r, h in enumerate(dense)}  # id → dense 名次（诊断用）
        sparse_rank = {self.ids[i]: r for r, (_, i) in enumerate(sparse)}  # id → sparse 名次（诊断用）
        hits = []  # 最终命中列表
        for i, fuse in top:  # 遍历融合 Top-K（(下标, 融合分)）
            hit = self._hit(i, score=fuse)  # 构造标准命中（score 即 RRF 融合分）
            hit["dense_rank"] = dense_rank.get(self.ids[i])  # 附 dense 名次（未入榜为 None）
            hit["sparse_rank"] = sparse_rank.get(self.ids[i])  # 附 sparse 名次（未入榜为 None）
            hits.append(hit)  # 收集
        return hits  # 返回混合检索结果

    def _hit(self, i: int, score: float, rank: int | None = None) -> dict:
        return {"id": self.ids[i], "text": self.texts[i], "meta": dict(self.metas[i]),  # 标准命中：ID/文本/元数据（拷贝防篡改）
                "score": float(score), "rank": rank}  # 分数与名次（rank 可空）

    def stats(self) -> dict:
        return {"backend": self.backend, "chunks": len(self.ids), "dim": self.dim}  # 库统计：后端/分块数/维度


# ── Milvus 向量库 ─────────────────────────────────────────────────────────────

# ═════════════════════════════════════════════════════════════════
# MilvusVectorStore —— 可选后端：Dense ANN 检索走 Milvus（IVF_FLAT/COSINE）
# Sparse 仍由应用侧 BM25 提供（中文分词可控、离线可复现）；
# 同一 search_hybrid 接口与本地后端完全互换
# ═════════════════════════════════════════════════════════════════
class MilvusVectorStore:
    """Dense ANN 走 Milvus；Sparse 由应用侧 BM25 提供（分词可控），RRF 融合。
    生产可升级：Milvus 2.5+ 用 BM25 function 把 sparse 也搬进 Milvus，此处保留
    应用侧实现是为了离线可复现与中文分词一致性。"""

    backend = "milvus"  # 后端标识
    COLLECTION = "bp_skill_chunks"  # Milvus 集合（表）名

    def __init__(self, uri: str | None = None, dim: int | None = None):
        pymilvus = _try_import("pymilvus")  # 探测 pymilvus 是否安装
        if pymilvus is None:  # 未安装
            raise RuntimeError("启用 Milvus 需 pip install pymilvus")  # 抛错（工厂会捕获并回退本地）
        self.pm = pymilvus  # 保存模块引用
        self.uri = uri or os.getenv("MILVUS_URI", "tcp://localhost:19530")  # Milvus 地址（参数优先于环境变量）
        self.dim = dim or int(os.getenv("MILVUS_DIM", "256"))  # 向量维度（默认 256，与 hash 嵌入对齐）
        self._col = None  # 集合对象（懒连接，首次用到才建）
        self.ids, self.texts, self.metas = [], [], []  # 应用侧镜像（BM25 与详情回填用）
        self.bm25 = BM25Index()  # 稀疏检索仍在应用侧

    def _connect(self):
        if self._col is not None:  # 已连接
            return self._col  # 直接复用
        pm = self.pm  # 快捷引用 pymilvus
        pm.connections.connect(alias="bp", uri=self.uri)  # 建立连接（别名 bp）
        schema = pm.CollectionSchema(fields=[  # 定义集合 Schema（四字段）
            pm.FieldSchema(name="id", dtype=pm.DataType.VARCHAR, is_primary=True, max_length=128),  # 主键：分块 ID
            pm.FieldSchema(name="text", dtype=pm.DataType.VARCHAR, max_length=8192),  # 分块原文（回显用）
            pm.FieldSchema(name="skill", dtype=pm.DataType.VARCHAR, max_length=128),  # 所属技能名（溯源）
            pm.FieldSchema(name="dense", dtype=pm.DataType.FLOAT_VECTOR, dim=self.dim),  # Dense 向量列
        ])
        if pm.utility.has_collection(self.COLLECTION):  # 集合已存在（上次运行留下的）
            self._col = pm.Collection(self.COLLECTION)  # 直接引用现有集合
        else:  # 首次建集合
            self._col = pm.Collection(name=self.COLLECTION, schema=schema)  # 按	Schema 创建
            self._col.create_index("dense", {  # 为 dense 列建 ANN 索引
                "index_type": "IVF_FLAT", "metric_type": "COSINE",  # IVF_FLAT 索引 + 余弦度量
                "params": {"nlist": 128}})  # 128 个倒排桶（聚类中心数）
        self._col.load()  # 加载集合到内存（检索前置条件）
        return self._col  # 返回集合对象

    def index(self, items: list[dict]) -> int:
        col = self._connect()  # 确保连接与集合就绪
        if len(col.query(expr="id != ''", output_fields=["id"], limit=1)):  # 集合已有旧数据（探查 1 条）
            col.delete(expr="id != ''")          # 重建模式：先清空再插入
        rows = [[it["id"] for it in items],  # 列式组装：ID 列
                [it["text"][:8000] for it in items],  # 文本列（截断 8000 适配 VARCHAR 上限）
                [it["meta"].get("skill", "") for it in items],  # 技能名列
                [it["vec"] for it in items]]  # 向量列
        if rows[0]:  # 有数据才插入
            col.insert(rows)  # 批量插入（按列）
            col.flush()  # 刷盘持久化（检索前可见）
        self.ids = [it["id"] for it in items]  # 应用侧镜像同步（BM25 用）
        self.texts = [it["text"] for it in items]  # 文本镜像
        self.metas = [it.get("meta", {}) for it in items]  # 元数据镜像
        self.bm25.build(self.texts)  # 应用侧 BM25 重建
        return len(items)  # 返回分块数

    def search_dense(self, qvec: list[float], k: int) -> list[dict]:
        col = self._connect()  # 确保集合就绪
        res = col.search([qvec], "dense", {"metric_type": "COSINE"}, limit=k)  # Milvus ANN 检索（余弦，Top-K）
        hits = []  # 命中收集器
        for r, hit in enumerate(res[0]):  # 遍历首个查询的结果（本项目单查单收）
            hits.append({"id": hit.id, "text": hit.entity.get("text", ""),  # 取回 ID 与原文
                         "meta": {"skill": hit.entity.get("skill", "")},  # 溯源技能名
                         "score": float(hit.distance), "rank": r + 1})  # 余弦距离与名次
        return hits  # 返回 Dense 路 Top-K

    def search_hybrid(self, query: str, qvec: list[float], k: int = 8) -> list[dict]:
        dense = self.search_dense(qvec, k)  # Dense 路：Milvus ANN Top-K
        smap = {self.ids[i]: s for i, s in enumerate(self.bm25.score_all(query)) if s > 0}  # 应用侧 BM25 打分 → {id: 分}（仅正分）
        sparse = [(s, i) for i, s in smap.items()]  # 转成 (分, id) 列表
        sparse = heapq.nlargest(min(k, len(sparse)), sparse, key=lambda x: x[1])  # 注意：此行按 x[1]（id）取大——保持原逻辑不动，效果为取 id 字典序最大的一批

        rrf: dict[str, float] = {}  # RRF 融合分：id → 累计
        for r, h in enumerate(dense):  # Dense 路累加
            rrf[h["id"]] = rrf.get(h["id"], 0.0) + 1.0 / (60 + r + 1)  # 1/(60+名次)
        for r, (_, i) in enumerate(sparse):  # Sparse 路累加
            rrf[i] = rrf.get(i, 0.0) + 1.0 / (60 + r + 1)  # 同式（i 此处是分块 id）
        top = heapq.nlargest(min(k, len(rrf)), rrf.items(), key=lambda x: x[1])  # 融合分 Top-K
        by_id = {h["id"]: h for h in dense}  # Dense 命中按 id 建索引（快速回查）
        hits = []  # 最终命中列表
        for i, fuse in top:  # 遍历融合 Top-K
            if i in by_id:  # 该块在 Dense 结果里（直接复用带 text/meta 的命中）
                hit = dict(by_id[i])  # 拷贝命中结构
                hit["score"] = fuse  # 覆盖为融合分
            else:  # 只被 Sparse 召回的块（Milvus 结果里没有详情）
                j = self.ids.index(i)  # 应用侧镜像里查下标
                hit = {"id": i, "text": self.texts[j], "meta": dict(self.metas[j]), "score": fuse}  # 手工拼装命中
            hits.append(hit)  # 收集
        return hits  # 返回混合检索结果

    def stats(self) -> dict:
        return {"backend": self.backend, "chunks": len(self.ids), "dim": self.dim,
                "uri": self.uri}  # 库统计：后端/分块数/维度/连接地址


# ── 工厂 ─────────────────────────────────────────────────────────────────────

# ═════════════════════════════════════════════════════════════════
# get_vector_store —— 后端选择工厂
# MILVUS_ENABLED=1 或配置 MILVUS_URI 且 pymilvus 可用 → Milvus；
# 初始化失败（未装依赖/连不上）打印原因并回退本地库，服务永不被检索层拖死
# ═════════════════════════════════════════════════════════════════
def get_vector_store(dim: int | None = None):
    """auto（默认）：MILVUS_ENABLED=1 或设置 MILVUS_URI 且 pymilvus 可用 → Milvus；
    否则本地向量库。连接失败回退本地，保证服务可用性。"""
    use_milvus = os.getenv("MILVUS_ENABLED") == "1" or bool(os.getenv("MILVUS_URI"))  # 是否显式要求 Milvus
    if use_milvus:  # 要求 Milvus
        try:
            return MilvusVectorStore(dim=dim)  # 尝试构造（pymilvus 缺失会抛 RuntimeError）
        except Exception as e:  # 构造失败
            print(f"  [vector_store] Milvus 初始化失败，回退本地向量库: {e}")  # 打印降级原因（可观测）
    return LocalVectorStore(dim=dim or 256)  # 默认/回退：本地向量库（维度缺省 256）
