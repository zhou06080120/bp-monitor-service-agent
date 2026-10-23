"""向量检索单元测试：BM25 稀疏检索、Dense 余弦检索、RRF 混合融合。

全部使用 HashEmbeddingProvider 造确定性向量，零外部依赖。
"""

from embeddings import HashEmbeddingProvider
from vector_store import BM25Index, LocalVectorStore

TEXTS = [
    "血压计袖带应绑在左臂上臂位置，与心脏保持同一水平。",
    "更换电池时请打开背盖，装入两节七号电池，注意正负极方向。",
    "打开手机蓝牙，在 App 中搜索设备并完成配对连接。",
]


def _build_store() -> LocalVectorStore:
    """用 3 条确定性文本建索引（向量由 hash 嵌入生成）。"""
    provider = HashEmbeddingProvider()
    store = LocalVectorStore(dim=provider.dim)
    items = [
        {"id": f"chunk-{i}", "text": t, "meta": {"skill": f"skill-{i}"},
         "vec": provider.embed_single(t)}
        for i, t in enumerate(TEXTS)
    ]
    store.index(items)
    return store


# ── BM25 ───────────────────────────────────────────────────────────────────────

def test_bm25_ranks_matching_doc_first():
    bm25 = BM25Index()
    bm25.build(TEXTS)

    scores = bm25.score_all("袖带绑在哪里")
    # 含"袖带/绑"等 token 的文档 0 应得最高分
    assert scores.index(max(scores)) == 0


def test_bm25_empty_query_returns_aligned_zeros():
    bm25 = BM25Index()
    bm25.build(TEXTS)

    scores = bm25.score_all("")
    # 返回与文档数等长的全零分（保持下标对齐）
    assert scores == [0.0] * len(TEXTS)


def test_bm25_no_match_all_zero():
    bm25 = BM25Index()
    bm25.build(TEXTS)

    # "量子"等字与库内三篇文档（袖带/电池/蓝牙）无任何 token 重叠
    scores = bm25.score_all("量子纠缠实验")
    assert max(scores) == 0.0


# ── Dense ──────────────────────────────────────────────────────────────────────

def test_dense_top1_is_self():
    store = _build_store()
    provider = HashEmbeddingProvider()

    hits = store.search_dense(provider.embed_single(TEXTS[1]), k=2)
    # 自查询：文本 1 的向量与库内自身余弦=1，必为第一名
    assert hits[0]["id"] == "chunk-1"
    assert hits[0]["rank"] == 1
    assert hits[0]["score"] > 0.99


def test_dense_empty_store_returns_empty():
    store = LocalVectorStore(dim=256)
    assert store.search_dense([0.1] * 256, k=3) == []


# ── 混合检索（RRF） ────────────────────────────────────────────────────────────

def test_hybrid_returns_k_hits_with_dual_ranks():
    store = _build_store()
    provider = HashEmbeddingProvider()

    hits = store.search_hybrid("袖带绑在哪里", provider.embed_single("袖带绑在哪里"), k=3)
    assert len(hits) == 3
    for h in hits:
        assert h["score"] > 0
        # 至少进入了一路榜单（双路名次用于诊断，未入榜为 None）
        assert h["dense_rank"] is not None or h["sparse_rank"] is not None
    # RRF 分单调递减
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_both_path_hit_ranks_first():
    """两路都命中的文档，RRF 累积分最高，应排第一。"""
    store = _build_store()
    provider = HashEmbeddingProvider()
    query = "血压计袖带绑在哪里"

    hits = store.search_hybrid(query, provider.embed_single(query), k=3)
    top = hits[0]
    # 文档 0 在 dense 与 sparse 两路都应上榜 → 融合分最高
    assert top["id"] == "chunk-0"
    assert top["dense_rank"] is not None
    assert top["sparse_rank"] is not None


def test_hybrid_k_larger_than_corpus_returns_all():
    store = _build_store()
    provider = HashEmbeddingProvider()

    hits = store.search_hybrid("电池", provider.embed_single("电池"), k=10)
    assert len(hits) <= 3  # 库容 3，k 超出时只返回全部


def test_stats_reports_backend_and_chunks():
    store = _build_store()
    stats = store.stats()
    assert stats == {"backend": "local", "chunks": 3, "dim": 256}
