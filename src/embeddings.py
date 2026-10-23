"""
嵌入层：Dense 语义向量 + 词法特征，供「双模态评估」与「Skill 向量检索」共用。

三级 Dense 后端（按环境变量自动选择，保证零配置也能跑）：
  1. hash    : 特征哈希向量（中文 char unigram+bigram → 512 维，L2 归一化）
               零依赖、确定性，离线可用；词法重叠近似语义，适合 Demo/降级
  2. local   : sentence-transformers 加载本地模型（如 bge-small-zh-v1.5）
              设置 BP_EMBED_MODEL_DIR 指向模型目录启用，无网络、有真实语义
  3. api     : OpenAI 兼容 Embedding API（DashScope / SiliconFlow 等）
              设置 EMBED_API_BASE / EMBED_API_KEY / EMBED_MODEL 启用

Sparse（BM25）编码在 vector_store.py 中实现（需基于全库统计 idf），
这里只提供统一的 tokenize 接口。

使用方式：
  from embeddings import get_embedding_provider
  embedder = get_embedding_provider()
  vec = embedder("袖带应该绑在哪只手臂上？")
"""

import os  # 读取后端选择与各后端配置的环境变量
import re  # 正则分词：抽取英文/数字整词与单个中文字符
import math  # L2 归一化所需的 sqrt 平方根计算
import zlib  # crc32 确定性哈希：把 token 映射到固定桶（跨进程可复现）


# ── 通用分词：中文连续字串 → unigram + bigram（含数字/英文整词） ────────────

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]")  # 预编译分词正则：连续字母数字整词 或 单个汉字


def tokenize(text: str) -> list[str]:
    """把中文/数字/英文转换成可检索的 token 序列（unigram + bigram）。"""
    tokens = _TOKEN_RE.findall(text.lower() if isinstance(text, str) else "")  # 小写化后抽取 token（非字符串按空文本处理）
    unigrams = list(tokens)  # 一元 token：单字/整词本身
    bigrams = [a + b for a, b in zip(tokens, tokens[1:])]  # 二元 token：相邻两个字/词拼接（补足中文词序语义）
    return unigrams + bigrams  # 两种粒度合并返回（供 hash 嵌入与 BM25 共用）


# ── Dense 1: 特征哈希（零依赖降级方案） ──────────────────────────────────────

# ═════════════════════════════════════════════════════════════════
# HashEmbeddingProvider —— 兜底嵌入后端
# token 经 crc32 哈希落入 256 维桶（带正负号计数），L2 归一化后即成向量；
# 语义能力弱但零依赖、确定性，保证离线环境检索/评估链路仍可运行
# ═════════════════════════════════════════════════════════════════
class HashEmbeddingProvider:
    """确定性特征哈希向量：token 落入 512 个桶，符号计数后 L2 归一化。
    无外部依赖，语义表达弱于真实模型，但零成本、可复现，适合离线 Demo。"""

    dim = 256  # 向量维度（类属性，检索侧据此建索引）
    name = "hash-256d"  # 后端名（日志/last_mount 展示用）

    def embed(self, texts: list[str]):
        return [self._embed_one(t) for t in texts]  # 批量嵌入：逐条调用单条实现

    def embed_single(self, text: str):
        return self._embed_one(text)  # 单条嵌入入口

    def _embed_one(self, text: str):
        vec = [0.0] * self.dim  # 初始化全零向量（各桶的累加计数器）
        for tok in tokenize(text):  # 遍历该文本的全部 token（unigram+bigram）
            # crc32 是跨进程确定性的（Python 内置 hash 带随机盐，不能用于索引）
            h = zlib.crc32(tok.encode("utf-8"))  # 对 token 做 crc32 哈希（同 token 永远同值）
            idx = h % self.dim  # 哈希取模得到落入的桶下标
            vec[idx] += 1.0 if (h >> 8) % 2 == 0 else -1.0  # 用哈希第 8 位的奇偶决定加减号（减少哈希碰撞的抵消）
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0  # 计算 L2 范数（全零向量时取 1 防除零）
        return [v / norm for v in vec]  # 归一化：余弦相似度只需点积即可


# ── Dense 2: 本地 sentence-transformers ──────────────────────────────────────

# ═════════════════════════════════════════════════════════════════
# SentenceTransformerProvider —— 本地真模型嵌入后端
# 懒加载：首次用到时才 import sentence_transformers 并载入模型，
# 无网络环境（BP_EMBED_MODEL_DIR 指向本地目录）也能拿到真实语义向量
# ═════════════════════════════════════════════════════════════════
class SentenceTransformerProvider:
    """本地语义模型后端（如 bge-small-zh-v1.5）。
    BP_EMBED_MODEL_DIR：模型目录路径；BP_EMBED_MODEL：HuggingFace 模型名。"""

    def __init__(self, model_dir: str | None, model_name: str):
        self.model_dir = model_dir  # 本地模型目录（None 则按模型名联网加载）
        self.model_name = model_name  # HuggingFace 模型名
        self._model = None  # 懒加载的模型实例（首次 embed 时才真正载入）

    @property
    def name(self):
        return f"local-ST({self.model_name})"  # 后端名含模型名，便于日志辨识

    @property
    def dim(self):
        self._load()  # 确保模型已加载（维度需问模型本身）
        return self._model.get_sentence_embedding_dimension()  # 返回模型输出维度

    def _load(self):
        if self._model is None:  # 尚未加载过
            from sentence_transformers import SentenceTransformer  # 延迟导入：不用本后端就不装依赖
            target = self.model_dir or self.model_name  # 优先本地目录，其次 HF 模型名
            self._model = SentenceTransformer(target)  # 载入模型（可能耗时数秒）
        return self._model  # 返回已缓存的模型实例

    def embed(self, texts: list[str]):
        model = self._load()  # 确保模型就绪
        vecs = model.encode(texts, normalize_embeddings=True).tolist()  # 批量编码并直接 L2 归一化，转 list
        return vecs  # 返回 list[list[float]]，与其他后端接口一致

    def embed_single(self, text: str):
        return self.embed([text])[0]  # 单条 = 批量一条后取首个


# ── Dense 3: OpenAI 兼容 Embedding API ───────────────────────────────────────

# ═════════════════════════════════════════════════════════════════
# OpenAICompatibleEmbeddingProvider —— 远程 API 嵌入后端
# 走 OpenAI 兼容协议（DashScope text-embedding-v3 等），语义质量最高；
# 维度从环境变量读取（不同模型的维度不同，需与检索库配置一致）
# ═════════════════════════════════════════════════════════════════
class OpenAICompatibleEmbeddingProvider:
    """OpenAI 兼容 Embedding API（DashScope text-embedding-v3 等）。
    环境变量：EMBED_API_BASE / EMBED_API_KEY / EMBED_MODEL。"""

    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url  # API 入口地址
        self.api_key = api_key  # 认证密钥
        self.model = model  # Embedding 模型名
        self._client = None  # 懒加载的 SDK 客户端

    @property
    def name(self):
        return f"api({self.model})"  # 后端名含模型名

    @property
    def dim(self):
        return int(os.getenv("EMBED_DIM", "1024"))  # 维度由环境变量指定（默认 1024，需与建库时一致）

    def _client_(self):
        if self._client is None:  # 客户端尚未创建
            from openai import OpenAI  # 延迟导入 SDK
            self._client = OpenAI(  # 构造兼容客户端
                api_key=self.api_key, base_url=self.base_url,  # 认证与入口
                timeout=float(os.getenv("LLM_TIMEOUT", "60")),  # 请求超时（与主 LLM 共用配置）
                max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),  # 自动重试次数
            )
        return self._client  # 返回缓存客户端

    def embed(self, texts: list[str]):
        client = self._client_()  # 拿到（或创建）客户端
        resp = client.embeddings.create(model=self.model, input=texts)  # 批量调用 Embedding API
        vecs = [d.embedding for d in resp.data]  # 从响应中抽出向量列表
        # L2 归一化（内积检索需要）
        return [[v / (math.sqrt(sum(x * x for x in v)) or 1.0) for v in vec] for vec in vecs]  # 每维除以范数（防零向量）
    def embed_single(self, text: str):
        return self.embed([text])[0]  # 单条 = 批量一条后取首个


# ── 工厂 ─────────────────────────────────────────────────────────────────────

# ═════════════════════════════════════════════════════════════════
# get_embedding_provider —— 三级降级工厂
# local → api → hash：auto 模式按环境可用性逐级尝试，全部不可用时
# 落到 hash 兜底，保证"零配置也能跑"；显式指定失败则直接抛错
# ═════════════════════════════════════════════════════════════════
def get_embedding_provider(force: str | None = None):
    """按 EMBED_PROVIDER 环境变量选择后端：
      'hash' | 'local' | 'api' | 'auto'（默认，按可用性逐级尝试）
    显式指定但不可用时抛异常；auto 时静默降级到 hash。"""
    choice = (force or os.getenv("EMBED_PROVIDER", "auto")).lower()  # 解析期望后端（参数优先于环境变量，统一小写）

    if choice in ("local", "auto"):  # local 显式要求，或 auto 顺位尝试本地模型
        model_dir = os.getenv("BP_EMBED_MODEL_DIR")  # 本地模型目录（None=未提供）
        model_name = os.getenv("BP_EMBED_MODEL", "BAAI/bge-small-zh-v1.5")  # 模型名（默认中文小模型）
        # auto 模式仅在显式给出模型目录时尝试本地（避免按 HF 名自动联网下载）；local 模式强制执行
        if model_dir or choice == "local":  # 有目录 或 显式 local 才尝试
            try:
                return SentenceTransformerProvider(model_dir=model_dir, model_name=model_name)  # 构造本地后端（真正加载延迟到首次 embed）
            except Exception as e:  # 构造失败（如依赖未安装）
                if choice == "local":  # 显式指定 local 却失败 → 不能静默降级
                    raise RuntimeError(f"本地嵌入模型加载失败: {e}") from e  # 抛出明确错误，带原始异常链

    if choice in ("api", "auto"):  # api 显式要求，或 auto 顺位尝试远程 API
        api_key = os.getenv("EMBED_API_KEY")  # 读取 API 密钥
        if api_key:  # 有密钥才可用 API 后端
            return OpenAICompatibleEmbeddingProvider(  # 构造 API 后端
                base_url=os.getenv("EMBED_API_BASE",  # API 入口，默认 DashScope 兼容模式
                                   "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                api_key=api_key,  # 认证密钥
                model=os.getenv("EMBED_MODEL", "text-embedding-v3"),  # 模型名，默认 DashScope v3
            )

    return HashEmbeddingProvider()  # 兜底：零依赖 hash 后端（auto 全失败或显式 hash）
