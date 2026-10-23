"""
评估器 v3：关键词规则 + 语义嵌入 + LLM-as-a-Judge 双模态联合校验。

三层架构（按成本/精度递进）：
  1. 规则层（硬闸，确定性）：
     - 推脱信号："联系人工" 一票否决（与 Agent 系统提示形成契约）
     - required 缺失 / forbidden 命中（含否定前置检测）
     医疗安全红线永远由规则层守住 —— Judge 无权翻案。
  2. 语义层（Embedding）：回答 vs 标准要点（note）的余弦相似度，
     用于识别"措辞不同但语义正确"的同义改写（如 3秒/三秒），
     零 LLM 成本，也可在无 Key 环境独立生效。
  3. Judge 层（LLM，JSON Mode）：仅裁决规则层与语义层冲突的模糊地带：
     - "缺少关键词"但语义相似度高 → Judge 判定 pass 可翻案（降低误判率）
     - 规则通过但 Judge 高置信度判 fail → 可翻案（降低漏判率）

模式（EVAL_MODE 或构造参数）：
  rule     仅规则层（原行为，完全可复现）
  semantic 规则 + 语义层记录，不翻案
  hybrid   规则 + 语义 + Judge 翻案（默认 auto：有 API Key 时启用）
  llm      Judge 单独裁决（实验对照用）

evaluate_answer 保持 (bool, reason) 两元组签名以兼容旧调用方；
每次判定的完整证据链写入 self.last_detail 与 self.stats。
"""

import os  # 读取评估模式与各阈值的覆盖环境变量
import json  # 加载评估集 JSON（60 题问答 + 判分要点）
import re  # 归一化数字千位分隔符；forbidden 否定前置检测的文本扫描
import logging  # 标准日志框架（print → logger 改造后的统一出口）
from pathlib import Path  # 读取评估集文件
from collections import defaultdict  # 分类统计的字典工厂（缺失键自动初始化）

from embeddings import get_embedding_provider  # 语义层嵌入（三级降级后端）
from llm_judge import LLMJudge  # 第三层：LLM 裁决器（JSON Mode + 多数投票）

logger = logging.getLogger(__name__)  # 模块 logger（经 setup_logging 统一格式化输出）

# ── 规则层常量 ───────────────────────────────────────────────────────────────

NEG_PREFIXES = ("不", "无", "非", "未", "没")  # 否定前缀词：forbidden 关键词前出现则不算命中（如"不含乳胶"）
NEG_WINDOW = 4  # 否定检测窗口：关键词前 4 个字符内找否定前缀
DEFERRAL_SIGNAL = "联系人工"  # 推脱信号：回答含此词即一票否决（与 Agent 系统提示的"不能答就推脱"契约对齐）

# ── 语义/Judge 阈值（可经环境变量覆盖，便于按嵌入后端/业务校准） ─────────────

SIM_CLEAR_PASS   = float(os.getenv("EVAL_SIM_CLEAR_PASS", "0.30"))   # 规则通过且相似度≥此值 → 直接通过
SIM_RESCUE_FLOOR = float(os.getenv("EVAL_SIM_RESCUE_FLOOR", "0.30")) # 规则失败但相似度≥此值 → 进 Judge
JUDGE_RESCUE_CONF    = float(os.getenv("EVAL_JUDGE_RESCUE_CONF", "0.70"))   # Judge 判 pass 救回误判的最小置信
JUDGE_OVERTURN_CONF  = float(os.getenv("EVAL_JUDGE_OVERTURN_CONF", "0.85")) # Judge 判 fail 推翻规则的最小置信


# ═════════════════════════════════════════════════════════════════
# _normalize：匹配前文本归一化——去掉数字间千位分隔符并小写化，
# 让 "22,000 次" 与 "22000 次" 在关键词匹配中等价
# ═════════════════════════════════════════════════════════════════
def _normalize(text: str) -> str:
    """匹配前归一化：数字千位分隔符一律去掉并小写化。"""
    return re.sub(r"(?<=\d)[,，](?=\d)", "", text).lower()  # 仅删除"前后都是数字"的逗号（中英文），再统一小写


# ═════════════════════════════════════════════════════════════════
# _forbidden_hits：forbidden 关键词是否"真正"出现
# 遍历全部出现位置，只要有一处不在否定语境（前 4 字符无否定词）即命中
# ═════════════════════════════════════════════════════════════════
def _forbidden_hits(text: str, keyword: str) -> bool:
    """forbidden 关键词是否"真正"出现（否定前置不算命中）。"""
    idx = 0  # 搜索起点（跳过已判定的出现位置）
    while True:  # 逐个出现位置检查
        pos = text.find(keyword, idx)  # 从 idx 起找下一个出现位置
        if pos == -1:  # 再无出现
            return False  # 全部出现位置都在否定语境 → 不算命中
        prefix = text[max(0, pos - NEG_WINDOW):pos]  # 取关键词前的窗口文本（防越界取 0）
        if not any(neg in prefix for neg in NEG_PREFIXES):  # 窗口内没有任何否定前缀
            return True  # 出现一处"真命中"即成立
        idx = pos + 1  # 此处是否定语境，跳过继续找下一处


# ═════════════════════════════════════════════════════════════════
# _cosine：余弦相似度。两侧向量都已 L2 归一化，点积即余弦值
# ═════════════════════════════════════════════════════════════════
def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))   # 两向量均已 L2 归一化，点积即余弦


# ═════════════════════════════════════════════════════════════════
# Evaluator —— 三层评估器（闭环的"考官"）
# 位置：每轮 run_baseline/run_block/run_final 都用它判分；失败样本
# (reason+detail) 是后台复盘器创建/修补 Skill 的直接输入
# ═════════════════════════════════════════════════════════════════
class Evaluator:
    def __init__(self, eval_set_path: str, mode: str = "auto",
                 judge: LLMJudge | None = None, embedder=None):
        data = json.loads(Path(eval_set_path).read_text(encoding="utf-8"))  # 读评估集 JSON
        self.questions = {q["id"]: q for q in data["questions"]}  # 题目按 id 建索引（qid → {question, ground_truth, note, category}）
        self.mode = self._resolve_mode(mode, judge)  # 解析实际运行模式（auto 按可用性降级）
        self.judge = judge or LLMJudge()  # Judge 裁决器（未注入则默认构造，无 Key 时 available=False）
        self.embedder = embedder or get_embedding_provider()  # 语义层嵌入器（未注入则三级降级选择）
        # Judge 多数投票次数（JUDGE_VOTE>1 时启用，降低单次裁决随机性）
        self.vote_n = max(1, int(os.getenv("JUDGE_VOTE", "1") or 1))  # 读取投票次数配置，至少 1 次
        self._judge_cache: dict[tuple[int, str], dict] = {}  # Judge 结果缓存（同题同答不重复调用，省成本）
        self.last_detail: dict = {}  # 最近一次判定的完整证据链（审计/复盘用）
        self.stats = {"calls": 0, "rule_only": 0, "judge_calls": 0,  # 评估统计：总调用/纯规则裁决数/Judge 调用数
                      "judge_rescues": 0, "judge_overturns": 0, "judge_errors": 0}  # Judge 救回数/推翻数/出错数

    def _resolve_mode(self, mode: str, judge: LLMJudge | None) -> str:
        if mode != "auto":  # 显式指定模式
            return mode  # 直接采用（rule/semantic/hybrid/llm 四选一）
        j = judge or LLMJudge()  # auto：探测 Judge 可用性（构造临时 Judge 或用传入的）
        return "hybrid" if j.available else "semantic"  # 有 Key → 混合三层；无 Key → 降级为规则+语义

    # ── 对外主接口 ────────────────────────────────────────────────────────────

    # ═══════════════════════════════════════════════════════════
    # evaluate_answer：单题判定主流程
    # 规则层 → 硬红线拦截 → 语义层 → （hybrid）模糊地带交 Judge 翻案
    # 推脱/禁止词是医疗安全红线，任何模式下都不可翻案
    # ═══════════════════════════════════════════════════════════
    def evaluate_answer(self, answer: str, question_id: int) -> tuple[bool, str]:
        """双模态联合判定，返回 (是否通过, 原因)。证据链见 self.last_detail。"""
        self.stats["calls"] += 1  # 判定调用计数
        q = self.questions[question_id]  # 取该题定义
        gt = q["ground_truth"]  # 判分要点（required/forbidden/note）
        question = q["question"]  # 题目原文（Judge 裁决需要）
        rule_ok, rule_reason = self._rule_check(answer, gt)  # 第一层：确定性规则硬闸

        detail = {"qid": question_id, "mode": self.mode,  # 证据链：题号与评估模式
                  "rule_ok": rule_ok, "rule_reason": rule_reason,  # 规则层结论与理由
                  "sim": None, "judge": None, "hard_fail": rule_reason.startswith(
                      "Agent 推脱") or rule_reason.startswith("出现禁止词")}  # 语义/Judge 占位；标记是否触碰硬红线

        # rule 模式：纯规则，完全可复现
        if self.mode == "rule":  # 显式 rule 模式
            self.last_detail = detail  # 存证据链
            self.stats["rule_only"] += 1  # 纯规则裁决计数
            return rule_ok, rule_reason  # 直接返回规则结论

        # 语义层：计算相似度。
        # 语义参照优先用题目级 note（"标准答案"全文），gt.note 只是兜底 ——
        # 因为 ground_truth 里只有零散关键词，直接拿来做参照会把"同义改写"
        # 误判成低相似度。
        ref = q.get("note") or gt.get("note") or " ".join(gt.get("required", []))  # 参照文本优先级：题目 note > gt note > required 词拼接
        sim = _cosine(self.embedder.embed_single(answer),  # 回答嵌入
                      self.embedder.embed_single(ref)) if ref else None  # 与参照嵌入算余弦；无参照则 None
        detail["sim"] = round(sim, 4) if sim is not None else None  # 记录相似度（4 位小数）

        # Judge 也一并带上完整语义参照（供裁决时比对要点）
        gt_judge = dict(gt)  # 复制一份要点（不污染原数据）
        if q.get("note"):  # 题目级 note 存在
            gt_judge["note"] = q["note"]  # 覆盖为更完整的参照文本
        detail["gt_ref"] = gt_judge  # 存入证据链（复盘时可见 Judge 看到的要点）

        # 硬红线（推脱/禁止词）无论任何模式都不翻案 —— 医疗安全
        if detail["hard_fail"]:  # 推脱或出现禁止词
            self.last_detail = detail  # 存证据链
            return False, rule_reason  # 无条件判错（Judge 无权翻案）

        if self.mode == "semantic":  # semantic 模式：规则+语义但不翻案
            self.last_detail = detail  # 存证据链
            self.stats["rule_only"] += 1  # 计数
            return rule_ok, rule_reason  # 仍按规则结论返回（sim 仅作记录）

        # llm/hybrid：Judge 层
        if self.mode == "llm":  # 实验对照模式：完全交给 Judge
            verdict_ok, verdict_reason = self._judge_path(question, gt_judge, answer, detail)  # Judge 单独裁决
            return verdict_ok, verdict_reason  # 返回 Judge 结论

        # hybrid：仅在模糊地带调用 Judge
        if rule_ok and (sim is None or sim >= SIM_CLEAR_PASS):  # 规则通过且语义清晰（或无参照）
            self.last_detail = detail          # 规则通过且语义清晰通过 → 无需 Judge
            self.stats["rule_only"] += 1  # 计数
            return True, rule_reason  # 直接通过（省一次 LLM 调用）
        if not rule_ok and not rule_reason.startswith("缺少关键词"):  # 规则失败但不是缺关键词（如推脱已在上面拦截，此处为兜底）
            self.last_detail = detail  # 存证据链
            self.stats["rule_only"] += 1  # 计数
            return False, rule_reason  # 直接判错（非模糊地带）
        if not rule_ok and (sim is not None and sim < SIM_RESCUE_FLOOR):  # 缺关键词且语义也相差甚远
            self.last_detail = detail          # 语义也不沾边 → 不值得问 Judge
            self.stats["rule_only"] += 1  # 计数
            return False, rule_reason  # 直接判错（真不会，翻案无意义）

        # 模糊地带：规则失败(缺关键词)但语义接近，或规则通过但语义存疑
        return self._judge_path(question, gt_judge, answer, detail)  # 交 Judge 裁决（可能双向翻案）

    # ═══════════════════════════════════════════════════════════
    # _judge_path：调用 Judge 裁决并按置信度融合（hybrid 双向翻案策略）
    # Judge 出错时回退规则结果（评估永不错挂）
    # ═══════════════════════════════════════════════════════════
    def _judge_path(self, question: str, gt: dict, answer: str,
                    detail: dict) -> tuple[bool, str]:
        """调用 LLM Judge 裁决，失败时回退规则结果。"""
        try:
            verdict = self._cached_judge(question, gt, answer)  # 走缓存的 Judge 裁决（含多数投票）
            self.stats["judge_calls"] += 1  # Judge 调用计数
        except Exception as e:  # Judge 调用失败（网络/Key 失效等）
            self.stats["judge_errors"] += 1  # 错误计数
            detail["judge"] = {"error": str(e)[:120]}  # 记录错误摘要
            self.last_detail = detail  # 存证据链
            return detail["rule_ok"], detail["rule_reason"]  # 回退规则结论

        detail["judge"] = verdict  # 记录 Judge 完整裁决
        self.last_detail = detail  # 存证据链
        v, conf = verdict["verdict"], verdict["confidence"]  # 取结论与置信度

        if self.mode == "llm":  # llm 模式：Judge 说了算
            ok = v == "pass"  # pass 才算对
            return ok, (f"judge: {verdict['reason']}" if ok
                        else f"judge fail: {verdict['reason']}")  # 返回带 judge 前缀的理由

        # hybrid 融合策略
        if v == "pass" and conf >= JUDGE_RESCUE_CONF and not detail["rule_ok"]:  # 规则判错但 Judge 高置信救回
            self.stats["judge_rescues"] += 1  # 救回计数（降低误判率）
            return True, f"语义符合（Judge翻案 conf={conf:.2f}）：{verdict['reason']}"  # 翻案为通过
        if v == "fail" and conf >= JUDGE_OVERTURN_CONF and detail["rule_ok"]:  # 规则通过但 Judge 高置信反对
            self.stats["judge_overturns"] += 1  # 推翻计数（降低漏判率；阈值更高=更谨慎）
            return False, f"语义偏离（Judge翻案 conf={conf:.2f}）：{verdict['reason']}"  # 翻案为失败
        self.stats["rule_only"] += 1  # 置信度不够/方向一致 → 维持规则结论
        return detail["rule_ok"], detail["rule_reason"]  # 不翻案

    def _cached_judge(self, question: str, gt: dict, answer: str) -> dict:
        key = (question, answer)  # 缓存键：同题同答结果必同（可复现 + 省钱）
        if key not in self._judge_cache:  # 缓存未命中
            if self.vote_n > 1 and hasattr(self.judge, "vote"):  # 配置了多数投票且 Judge 支持投票
                self._judge_cache[key] = self.judge.vote(question, gt, answer, n=self.vote_n)  # n 次采样投票
            else:
                self._judge_cache[key] = self.judge.judge(question, gt, answer)  # 单次裁决
        return self._judge_cache[key]  # 返回缓存/新算的裁决

    # ── 规则层（与旧版一致，保持可复现） ──────────────────────────────────────

    # ═══════════════════════════════════════════════════════════
    # _rule_check：确定性规则硬闸——推脱一票否决 → required 全含 →
    # forbidden 零命中（否定语境豁免）。返回 (结论, 机器可读理由)
    # ═══════════════════════════════════════════════════════════
    def _rule_check(self, answer: str, gt: dict) -> tuple[bool, str]:
        ans_norm = _normalize(answer)  # 回答归一化（千位分隔符/大小写）
        if DEFERRAL_SIGNAL in ans_norm:  # 推脱信号命中
            return False, f"Agent 推脱（含 '{DEFERRAL_SIGNAL}'）"  # 一票否决（硬红线）
        for kw in gt.get("required", []):  # 逐个检查必含关键词
            if _normalize(kw) not in ans_norm:  # 归一化后仍不包含
                return False, f"缺少关键词: '{kw}'"  # 判错并给出缺失词（复盘线索）
        for kw in gt.get("forbidden", []):  # 逐个检查禁止关键词
            if _forbidden_hits(ans_norm, _normalize(kw)):  # 真命中（否定语境不算）
                return False, f"出现禁止词: '{kw}'"  # 判错（硬红线）
        return True, "correct"  # 全部通过

    # ── 全量评估（兼容旧接口） ────────────────────────────────────────────────

    # ═══════════════════════════════════════════════════════════
    # run_full_eval：跑完整套评估集——逐题调用 agent_func 生成回答再判分，
    # 汇总总体/分类准确率与错误样本（错误样本即复盘器的输入）
    # ═══════════════════════════════════════════════════════════
    def run_full_eval(self, agent_func, verbose: bool = False) -> dict:
        total = correct = 0  # 总题数与答对数
        by_category = defaultdict(lambda: {"total": 0, "correct": 0})  # 分类统计桶（缺失自动建）
        errors = []  # 错误样本明细（复盘输入）
        for qid, q in sorted(self.questions.items()):  # 按题号顺序逐题评估
            answer = agent_func(q["question"])  # 调用被测 Agent 生成回答
            ok, reason = self.evaluate_answer(answer, qid)  # 三层联合判分
            total += 1  # 总数 +1
            by_category[q["category"]]["total"] += 1  # 类目总数 +1
            if ok:  # 答对
                correct += 1  # 对数 +1
                by_category[q["category"]]["correct"] += 1  # 类目对数 +1
            else:  # 答错
                errors.append({"id": qid, "category": q["category"], "reason": reason,  # 记录题号/类目/判错理由
                               "question": q["question"][:40],  # 问题摘要
                               "detail": self.last_detail})  # 完整证据链（含 sim/judge）
                if verbose:  # 详细模式
                    logger.info(f"✗ Q{qid} [{q['category']}] {reason}")  # 逐条记录失败信息
        for cat in by_category.values():  # 计算各分类准确率
            cat["accuracy"] = round(cat["correct"] / cat["total"], 3)  # 3 位小数
        return {"total": total, "correct": correct,  # 汇总结果
                "accuracy": round(correct / total, 3),  # 总体准确率（33%→93% 的度量口径）
                "by_category": dict(by_category), "errors": errors}  # 分类明细与错误样本

    def print_report(self, result: dict, label: str = ""):
        header = f"=== 评估报告 {label} ===" if label else "=== 评估报告 ==="  # 报告标题（可带标签如 baseline/final）
        logger.info(header)  # 记录标题
        logger.info(f"总体准确率: {result['correct']}/{result['total']} = {result['accuracy']:.1%}")  # 记录总体准确率
        logger.info("分类准确率:")  # 分节标题
        for cat, stats in sorted(result["by_category"].items()):  # 按类目名排序输出
            bar = "█" * int(stats["accuracy"] * 20)  # 文本柱状图：精度×20 个方块
            logger.info(f"  {cat:<20} {stats['correct']:>2}/{stats['total']:>2}  {bar} {stats['accuracy']:.0%}")  # 对齐记录类目/对错数/柱图/百分比
        logger.info(f"评估器统计: {self.stats}")  # 记录三层评估的调用统计（Judge 用量等）
        if result.get("errors"):  # 有错误样本
            logger.info("错误样本（前5条）:")  # 分节标题
            for err in result["errors"][:5]:  # 只展示前 5 条
                logger.info(f"  Q{err['id']} {err['question']}... → {err['reason']}")  # 记录题号/问题摘要/判错理由
