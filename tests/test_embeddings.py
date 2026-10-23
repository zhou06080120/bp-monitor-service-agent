"""嵌入层单元测试：分词器、hash 嵌入的确定性、工厂降级链。"""

import math

from embeddings import (
    HashEmbeddingProvider,
    get_embedding_provider,
    tokenize,
)


# ── tokenize ───────────────────────────────────────────────────────────────────

def test_tokenize_chinese_unigram_and_bigram():
    tokens = tokenize("血压计")
    # 单字 unigram
    for ch in ["血", "压", "计"]:
        assert ch in tokens
    # 相邻 bigram（补足中文词序语义）
    assert "血压" in tokens
    assert "压计" in tokens


def test_tokenize_keeps_alnum_words_whole():
    tokens = tokenize("型号 BP-100 血压计")
    # 英文/数字连续串作为整词（型号、序列号检索的关键行为）
    assert "bp" in tokens
    assert "100" in tokens


# ── HashEmbeddingProvider ──────────────────────────────────────────────────────

def test_hash_embedding_deterministic():
    """确定性是跨进程可复现索引的前提（crc32 而非带随机盐的 hash）。"""
    p = HashEmbeddingProvider()
    a = p.embed_single("袖带绑在左臂")
    b = p.embed_single("袖带绑在左臂")
    assert a == b


def test_hash_embedding_l2_normalized():
    p = HashEmbeddingProvider()
    vec = p.embed_single("蓝牙连接失败怎么办")
    norm = math.sqrt(sum(v * v for v in vec))
    assert abs(norm - 1.0) < 1e-9  # 归一化后余弦 = 点积


def test_hash_embedding_dim_matches_class_attr():
    p = HashEmbeddingProvider()
    assert len(p.embed_single("任意文本")) == p.dim == 256


def test_hash_embedding_distinguishes_texts():
    p = HashEmbeddingProvider()
    a = p.embed_single("血压计袖带位置")
    b = p.embed_single("蓝牙配对失败")
    assert a != b


def test_hash_embedding_batch_interface():
    p = HashEmbeddingProvider()
    vecs = p.embed(["文本一", "文本二"])
    assert len(vecs) == 2
    assert all(len(v) == p.dim for v in vecs)


# ── 工厂降级链 ─────────────────────────────────────────────────────────────────

def test_factory_explicit_hash():
    p = get_embedding_provider("hash")
    assert isinstance(p, HashEmbeddingProvider)


def test_factory_auto_without_key_falls_back_to_hash(monkeypatch):
    """零配置环境（无模型目录、无 API Key）：auto 必须落到 hash 兜底。"""
    monkeypatch.setenv("EMBED_PROVIDER", "auto")
    monkeypatch.delenv("BP_EMBED_MODEL_DIR", raising=False)
    monkeypatch.delenv("EMBED_API_KEY", raising=False)

    p = get_embedding_provider()
    assert isinstance(p, HashEmbeddingProvider)
