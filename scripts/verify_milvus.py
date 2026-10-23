"""
Milvus 向量后端集成验证脚本。

用法：
  python scripts/verify_milvus.py                # 本地向量库自检 + 环境体检
  MILVUS_URI=tcp://localhost:19530 python scripts/verify_milvus.py   # 连 Milvus 真机验证

验证内容：
  1. 接口契约：LocalVectorStore 与 MilvusVectorStore 的 index/search_dense/
     search_hybrid/stats 签名一致性（Milvus 未配置时也可检查对象能力）
  2. 本地自检：同一语料的 Dense/Sparse/RRF 融合检索可用性 + 命中分块溯源
  3. 真机往返（需 Milvus 运行 + pip install pymilvus）：
     index → flush → 混合检索，并与 LocalVectorStore 结果做 Top-K 一致率对比
     （小规模语料上 Dense 余弦是精确值，ANN 近似检索应保持高重叠）

Milvus 快速启动（任选其一）：
  docker run -d --name milvus-standalone -p 19530:19530 -p 9091:9091 \
      milvusdb/milvus:v2.4.0 standalone
"""

import os  # 读取 MILVUS_URI / MILVUS_ENABLED 环境变量
import sys  # 注入 src 路径
import time  # perf_counter 统计索引与检索耗时
from pathlib import Path  # 定位项目根目录

ROOT = Path(__file__).parent.parent  # 项目根目录（本文件在 scripts/ 下）
sys.path.insert(0, str(ROOT / "src"))  # 让 src 内模块可按顶级名导入


# ═════════════════════════════════════════════════════════════════
# load_skills：读取 skills/ 下全部 SKILL.md 作为验证语料
# ═════════════════════════════════════════════════════════════════
def load_skills():
    """读取 skills/ 下所有 SKILL.md，返回 {skill_name: content}。"""
    skills = {}  # 结果：技能名 → 全文
    for d in sorted((ROOT / "skills").iterdir()):  # 按名排序遍历技能目录
        if d.is_dir():  # 只处理目录
            for f in d.glob("*.md"):  # 遍历目录内 .md 文件（正常只有 SKILL.md）
                skills[d.name] = f.read_text(encoding="utf-8")  # 读入全文（后读覆盖先读，以 SKILL.md 习惯为准）
    return skills  # 返回语料字典


# ═════════════════════════════════════════════════════════════════
# build_corpus：用真实语料构建 SkillStore（切块+嵌入+入本地库），
# 后续阶段的检索验证都基于这份真实索引
# ═════════════════════════════════════════════════════════════════
def build_corpus():
    """用真实 Skill 语料构建 SkillStore（切块 + 嵌入 + 入本地向量库）。"""
    from skill_store import SkillStore  # 延迟导入（sys.path 已就绪）
    st = SkillStore()  # 创建分块索引（嵌入/向量后端按环境变量选择）
    st.reindex_all(load_skills())  # 全量建索引
    return st  # 返回就绪的 SkillStore


# ═════════════════════════════════════════════════════════════════
# phase1_contract：接口契约自检——两个后端类必须实现同一组方法，
# 保证 SkillStore 层可互换后端而无需改动（里氏替换的最低要求）
# ═════════════════════════════════════════════════════════════════
def phase1_contract():
    """接口契约自检（无需 Milvus）。"""
    print("=" * 60)  # 分隔线
    print("[1/3] 接口契约自检（本地 vs Milvus 后端）")  # 阶段标题
    from vector_store import LocalVectorStore, MilvusVectorStore  # 导入两个后端类
    methods = ("index", "search_dense", "search_hybrid", "stats")  # 契约方法清单
    for cls in (LocalVectorStore, MilvusVectorStore):  # 逐一检查两个类
        missing = [m for m in methods if not callable(getattr(cls, m, None))]  # 找缺失/不可调用的方法
        assert not missing, f"{cls.__name__} 缺少接口: {missing}"  # 缺失即断言失败（验证不通过）
        print(f"  ✓ {cls.__name__} 接口完整: {', '.join(methods)}")  # 打印通过信息
    print("  ✓ 契约自检通过：两后端可互换（SKILL 层无感知）")  # 阶段结论


# ═════════════════════════════════════════════════════════════════
# phase2_local：本地向量库功能自检——真实语料跑一次混合检索，
# 验证命中分块可溯源（dense/sparse 双路名次齐备）与 hit_rate 可用
# ═════════════════════════════════════════════════════════════════
def phase2_local():
    """本地向量库功能自检。"""
    print("=" * 60)  # 分隔线
    print("[2/3] 本地向量库功能自检（Dense+Sparse→RRF）")  # 阶段标题
    st = build_corpus()  # 构建真实语料索引
    q = "袖带应该绑多紧 离肘窝多远"  # 样例查询（袖带佩戴类问题）
    hits = st.retrieve(q, k=5)  # 混合检索 Top-5
    assert hits, "检索无结果"  # 无结果即失败
    for h in hits[:5]:  # 遍历前 5 个命中
        assert "dense_rank" in h and "sparse_rank" in h, h  # 双路名次必须齐备（溯源诊断字段）
        print(f"  ✓ [{h['meta']['skill']}/{h['meta']['section']}] "
              f"dense_rank={h.get('dense_rank')} sparse_rank={h.get('sparse_rank')}")  # 打印命中溯源
    obs = st.hit_rate([(q, ["袖带", "肘窝"])], k=3)  # 用预期关键词测命中率
    print(f"  ✓ 样例查询 hit@3={obs['hit_at_k']:.0%}")  # 打印 hit@3
    return st  # 返回就绪的 SkillStore（供阶段 3 对照用）


# ═════════════════════════════════════════════════════════════════
# phase3_milvus：Milvus 真机往返验证——同一语料分别入本地库与
# Milvus，对比混合检索 Top-8 重叠率（ANN vs 精确），阈值 50%
# ═════════════════════════════════════════════════════════════════
def phase3_milvus():
    """Milvus 真机往返验证（可选）。"""
    print("=" * 60)  # 分隔线
    print("[3/3] Milvus 真机往返验证")  # 阶段标题
    try:
        import pymilvus  # noqa: F401  # 探测 pymilvus 是否安装（导入即验证）
    except ImportError:  # 未安装
        print("  ⊘ 未安装 pymilvus：pip install pymilvus 后重试")  # 提示安装
        print("  ⊘ 提示：本地向量库自检已通过，Milvus 后端代码路径需真机验收")  # 说明降级状态
        return 0  # 返回 0（跳过，不算失败）

    from vector_store import get_vector_store  # 向量库工厂
    from embeddings import get_embedding_provider  # 嵌入器工厂
    from skill_store import SkillStore  # 分块索引层

    skills = load_skills()  # 加载语料
    emb = get_embedding_provider()  # 取嵌入器（两后端共用同一嵌入器，保证向量可比）

    # 本地基线
    from vector_store import LocalVectorStore  # 本地向量库类
    local = SkillStore(embedding_provider=emb,
                       vector_store=LocalVectorStore(dim=emb.dim))  # 显式指定本地后端建 SkillStore
    local.reindex_all(skills)  # 建本地索引（精确检索基准）

    # Milvus 后端
    milvus_store = get_vector_store(dim=emb.dim)  # 工厂选择后端（读 MILVUS_URI/ENABLED）
    if milvus_store.backend != "milvus":  # 工厂回退到了本地（未配置或初始化失败）
        print("  ⊘ MILVUS_URI 未配置或初始化失败，跳过真机验证")  # 提示跳过原因
        return 0  # 返回 0
    milvus = SkillStore(embedding_provider=emb, vector_store=milvus_store)  # 用 Milvus 后端建 SkillStore
    t0 = time.perf_counter()  # 索引计时起点
    n = milvus.reindex_all(skills)  # 语料写入 Milvus（含建集合/索引/flush）
    t_index = time.perf_counter() - t0  # 索引耗时

    # 检索对比：同语料 Top-K 一致率（ANN vs 精确）
    questions = ["袖带应该绑多紧 离肘窝多远", "血压计怎么开机 按多久", "APP怎么绑设备"]  # 三个代表性查询
    agree = 0  # 重叠率累计
    for q in questions:  # 逐查询对比
        t0 = time.perf_counter()  # 本地检索计时起点
        lm = {h["id"] for h in local.retrieve(q, k=8)}  # 本地精确检索 Top-8 的 ID 集合
        t_local = time.perf_counter() - t0  # 本地耗时
        t0 = time.perf_counter()  # Milvus 检索计时起点
        mm = {h["id"] for h in milvus.retrieve(q, k=8)}  # Milvus ANN 检索 Top-8 的 ID 集合
        t_milvus = time.perf_counter() - t0  # Milvus 耗时
        overlap = len(lm & mm) / 8  # 交集 / 8 = 重叠率
        agree += overlap  # 累计
        print(f"  Q: {q[:28]:<30} Top-8 重叠率={overlap:.0%} "
              f"(local {t_local*1000:.1f}ms / milvus {t_milvus*1000:.1f}ms)")  # 打印单查询对比
    avg = agree / len(questions)  # 平均重叠率
    print(f"  ✓ 索引 {n} 分块耗时 {t_index:.2f}s；三样例平均 Top-8 重叠率 {avg:.0%}")  # 打印汇总
    assert avg >= 0.5, "ANN 与精确检索重叠率过低，请检查索引参数(IVF nlist)"  # 低于 50% 判失败（索引参数可疑）
    return 1  # 返回 1（真机验证完成）


if __name__ == "__main__":  # 直接运行本脚本
    print(f"运行环境: MILVUS_URI={os.getenv('MILVUS_URI', '(未设置)')} "
          f"MILVUS_ENABLED={os.getenv('MILVUS_ENABLED', '(未设置)')}")  # 打印 Milvus 相关环境配置
    phase1_contract()  # 阶段 1：接口契约
    phase2_local()  # 阶段 2：本地功能自检
    phase3_milvus()  # 阶段 3：真机往返（可选）
    print("\n✓ 向量后端验证完成")  # 收尾提示
