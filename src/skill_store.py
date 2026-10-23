"""
Skill 检索挂载层：把细粒度 Skill 规则向量化入库，按需召回 Top-K 注入上下文。

解决的问题（业务几百个细分场景时）：
  把所有 Skill 全文塞进 Prompt → Token 爆炸 + 长文本注意力稀释。
  改为：提问时双向检索（Dense 语义 + Sparse 词汇 → RRF 融合）召回
  最相关的 Top-K 规则分块，动态注入系统提示，Token 成本收敛到常数级。

分块策略：
  按 Markdown 标题（## / ###）切分 SKILL.md，frontmatter 独立成块；
  每块标注 {skill, section}，检索命中后可溯源到具体规则小节。

索引时机：
  - 服务启动时全量索引
  - Skill 每次 create/patch 后增量重建（本次语料小，直接全量重建，毫秒级）
"""

import re  # 标题正则分块 + frontmatter 提取
import hashlib  # 生成分块稳定 ID（md5 截断），索引更新时可幂等覆盖
import logging  # 标准日志框架（print → logger 改造后的统一出口）

from embeddings import get_embedding_provider  # Dense 嵌入后端工厂（hash/local/api 三级降级）
from vector_store import get_vector_store  # 向量库工厂（本地 or Milvus，Dense+BM25+RRF 混合检索）

logger = logging.getLogger(__name__)  # 模块 logger（经 setup_logging 统一格式化输出）

_HEADING_RE = re.compile(r"^(#{2,4})\s+(.+)$", re.MULTILINE)  # 匹配 2~4 级 Markdown 标题行（##/###/####）


# ═════════════════════════════════════════════════════════════════
# SkillStore —— Skill 分块索引 + 混合检索（闭环的知识供给端）
# Agent 侧 skill 多到超预算时，经 retrieve() 召回 Top-K 相关分块
# 注入系统提示；hit_rate() 则评估检索器自身对评估集的召回能力
# ═════════════════════════════════════════════════════════════════
class SkillStore:
    def __init__(self, embedding_provider=None, vector_store=None):
        self.embed = embedding_provider or get_embedding_provider()  # 嵌入器：未注入则按环境变量三级降级选择
        self.store = vector_store or get_vector_store(dim=getattr(self.embed, "dim", 256))  # 向量库：维度取嵌入器 dim（无该属性则 256）
        self.chunks: list[dict] = []  # 当前已索引的全部分块（含 id/text/meta）

    # ── 分块 ─────────────────────────────────────────────────────────────────

    # ═══════════════════════════════════════════════════════════
    # chunk_skill：按 Markdown 标题把单个 SKILL.md 切成可检索小节块
    # frontmatter（--- ... ---）单独成块标记 _meta；小节路径含父标题便于溯源
    # ═══════════════════════════════════════════════════════════
    @staticmethod
    def chunk_skill(skill_name: str, content: str) -> list[dict]:
        """按标题把 SKILL.md 切成 {id, text, meta:{skill, section}} 块。
        frontmatter（--- 之间）单列一块，section 记为 '_meta'。"""
        chunks = []  # 收集该技能的全部分块
        fm_m = re.match(r"^---\n(.*?)\n---\n", content, re.DOTALL)  # 尝试从文件头匹配 frontmatter 块
        offset = fm_m.end() if fm_m else 0  # 正文起点：frontmatter 之后；无则从 0 开始
        if fm_m:  # 存在 frontmatter（技能元信息：名称/描述/版本）
            chunks.append({"skill": skill_name, "section": "_meta",  # 元信息块，小节名固定 _meta
                           "text": fm_m.group(1).strip()[:2000]})  # 内容截断 2000 字

        # 扫描全文标题：记录每个标题的 (起始位置, 标题文本, 井号级数)
        positions = [(m.start(), m.group(2).strip(), m.group(1).count("#"))
                     for m in _HEADING_RE.finditer(content, offset)]  # 从正文起点开始找标题
        for idx, (start, title, level) in enumerate(positions):  # 逐标题切出一个小节块
            end = positions[idx + 1][0] if idx + 1 < len(positions) else len(content)  # 块终点 = 下一个标题起点（最后一块到文件尾）
            text = content[start:end].strip()  # 该小节全文（含标题行）
            if not text:  # 空块（如连续标题）
                continue  # 跳过
            parent = ""  # 向上寻找最近的更高级标题作为父级（构建 "父/子" 路径）
            for (s, t, lv) in reversed(positions[:idx]):  # 从当前标题之前倒序找
                if lv < level:  # 级数更小 = 层级更高
                    parent = t  # 记下父标题名
                    break  # 找到最近的即可
            chunks.append({"skill": skill_name,  # 所属技能名
                           "section": f"{parent}/{title}" if parent else title,  # 小节路径：有父级则 "父/子"
                           "text": text[:3000]})  # 单块截断 3000 字防超长

        if not chunks:  # 无标题 → 整篇一块
            body = content[offset:].strip()  # 取 frontmatter 之后的正文
            if body:  # 正文非空才入库
                chunks.append({"skill": skill_name, "section": "_all", "text": body[:3000]})  # 小节名固定 _all
        return chunks  # 返回该技能的分块列表

    # ── 索引 ─────────────────────────────────────────────────────────────────

    # ═══════════════════════════════════════════════════════════
    # reindex_all：全量重建索引（切块 → 批量嵌入 → 写入向量库）
    # Skill create/patch 后调用，进化出的新规则即时可被检索
    # ═══════════════════════════════════════════════════════════
    def reindex_all(self, skills: dict[str, str]) -> int:
        """skills: {name: SKILL.md content}。全量切块 + 编码 + 入库。"""
        self.chunks = []  # 清空旧分块，重新累计
        for name, content in sorted(skills.items()):  # 按技能名排序保证重建顺序稳定
            for c in self.chunk_skill(name, content):  # 该技能的每个分块
                cid = hashlib.md5(f"{name}|{c['section']}".encode()).hexdigest()[:24]  # 用"技能名|小节名"生成稳定 ID（重建可幂等覆盖）
                self.chunks.append({"id": cid, "text": c["text"],  # 入分块表
                                    "meta": {"skill": c["skill"], "section": c["section"]}})  # 溯源元数据
        items = []  # 待写入向量库的完整条目（含向量）
        if self.chunks:  # 有分块才需要嵌入
            vecs = self.embed.embed([c["text"] for c in self.chunks])  # 批量嵌入全部分块文本
            items = [{"id": c["id"], "text": c["text"], "meta": c["meta"], "vec": v}
                     for c, v in zip(self.chunks, vecs)]  # 分块与向量一一配对组装
        n = self.store.index(items)  # 写入向量库（本地/Milvus 统一接口），返回条数
        logger.info(f"[SkillStore] 已索引 {n} 个规则分块（backend={self.store.backend}, "
                    f"embedder={self.embed.name}）")  # 记录索引摘要（后端与嵌入器名，便于确认运行形态）
        return n  # 返回索引分块数

    def reindex_skill(self, skill_name: str, content: str, all_skills: dict[str, str]):
        """Skill 变更后的增量入口：直接全量重建（当前规模毫秒级）。"""
        return self.reindex_all(all_skills)  # 语料小 → 简化为全量重建，避免增量同步的复杂度

    # ── 检索 ─────────────────────────────────────────────────────────────────

    # ═══════════════════════════════════════════════════════════
    # retrieve：Agent 侧核心入口——混合检索 + 字符预算打包
    # 先多召回一些（max(k,10)），再做同小节去重与预算裁剪
    # ═══════════════════════════════════════════════════════════
    def retrieve(self, query: str, k: int = 8, budget_chars: int = 4000) -> list[dict]:
        """混合检索 Top-K 规则分块，并按字符预算打包（剔除同源重复块）。"""
        qvec = self.embed.embed_single(query)  # 把问题编码成查询向量
        hits = self.store.search_hybrid(query, qvec, k=max(k, 10))  # 混合检索（Dense+BM25+RRF），多召回防去重后不足
        picked, used, seen_sections = [], 0, set()  # 已选块 / 已用字符数 / 已见小节集合
        for h in hits:  # 按融合相关性从高到低遍历候选
            sec_key = f"{h['meta'].get('skill')}|{h['meta'].get('section')}"  # 以"技能|小节"作为去重键
            if sec_key in seen_sections:  # 同一小节已选过
                continue                      # 同一小节只取一次，防止重复灌入
            seen_sections.add(sec_key)  # 标记该小节已处理
            if used + len(h["text"]) > budget_chars:  # 加入此块会超出字符预算
                continue  # 跳过该块（继续尝试更短的后续块）
            used += len(h["text"])  # 累加已用字符
            picked.append(h)  # 选中该块
            if len(picked) >= k:  # 已选满 K 块
                break  # 提前结束
        return picked  # 返回最终挂载的分块列表

    # ═══════════════════════════════════════════════════════════
    # hit_rate：检索器自身的可观测性——对已知问题评估 Top-K 召回
    # 输出 hit@1 / hit@k / MRR 与分题明细，用于验证"检索挂载是否找得到对的规则"
    # ═══════════════════════════════════════════════════════════
    def hit_rate(self, queries: list[tuple[str, list[str]]],
                 k: int = 3) -> dict:
        """检索可观测性：评估混合检索对已知问题的召回能力。

        queries: [(question, expected_keywords)] —— expected_keywords 为该题
                 标准要点关键词（如 eval_set 的 required 词）。
        命中定义：Top-K 分块文本中出现任一预期关键词。
        返回 hit@{top1/top3}、MRR、分题明细。"""
        exact, within_k, mrr_sum, details = 0, 0, 0.0, []  # hit@1 计数 / hit@k 计数 / MRR 累加 / 分题明细
        for i, (query, kws) in enumerate(queries):  # 逐题评估（i 预留题号）
            hits = self.retrieve(query, k=k, budget_chars=4000)  # 用与线上一致的检索入口取 Top-K
            hit_rank = None  # 首个命中关键词的排名（None=全部未命中）
            for r, h in enumerate(hits, 1):  # 按排名遍历命中块（排名从 1 起）
                if any(kw in h["text"] for kw in kws):  # 块文本包含任一预期关键词
                    hit_rank = r  # 记录首次命中排名
                    break  # 只关心第一个
            if hit_rank == 1:  # 第一名即命中
                exact += 1  # hit@1 计数 +1
            if hit_rank is not None:  # Top-K 内命中
                within_k += 1  # hit@k 计数 +1
                mrr_sum += 1.0 / hit_rank  # MRR 累加倒数排名
            details.append({"query": query[:40], "hit_rank": hit_rank,  # 记录题目摘要与命中排名
                            "top_sections": [f"{h['meta']['skill']}/{h['meta']['section']}"
                                             for h in hits[:3]]})  # Top-3 分块溯源（排查用）
        n = len(queries) or 1  # 题数（0 时取 1 防除零）
        return {"total": len(queries), "hit_at_1": round(exact / n, 3),  # 总题数与 hit@1
                "hit_at_k": round(within_k / n, 3), "k": k,  # hit@k 与评估用的 K 值
                "mrr": round(mrr_sum / n, 3), "details": details}  # MRR 与分题明细

    def stats(self) -> dict:
        s = dict(self.store.stats())  # 复制向量库统计（条数/后端等）
        s["embedder"] = self.embed.name  # 补充嵌入器名
        s["chunked_skills"] = len({c["meta"]["skill"] for c in self.chunks})  # 补充已分块的技能数（去重统计）
        return s  # 返回完整统计快照
