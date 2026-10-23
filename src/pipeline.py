"""
实验 Pipeline：所有耗时任务（基线评估 / 块进化 / 最终评估）的纯函数化实现。

被 task_queue 的后台 Worker 调用（in-process 线程或 Celery 进程均可），
与 HTTP 层完全解耦：
  - 实验状态持久化到 outputs/experiment_state.json（跨进程可见）
  - 进度经 emit(type, **payload) 回调写入 job 事件日志
  - 组件（SkillManager/Evaluator/SkillStore/Agent/Reviewer）按进程懒加载缓存

事件类型与旧版 serve.py 流式接口保持一致（eval_q/eval_complete/.../phase_change），
保证前端零改动兼容；新增：
  - eval_q / question_result 附 mount（向量检索挂载信息）与 judge（双模态判定证据）
  - eval_complete 附 judge_stats（Judge 翻案统计）与 business_metrics（终局）
  - retrieval_info（Skill 变更后的重建索引事件）
"""

import os  # 读取 EVAL_MODE / EVAL_GATE_MARGIN 等环境变量
import json  # 实验状态与评估结果的 JSON 读写
import time  # perf_counter 计算单题端到端延迟
from pathlib import Path  # 统一管理 skills/outputs/data 各目录路径
from datetime import datetime  # 评估结果打时间戳

from redline_guard import check_action as redline_check_action  # 安全红线守卫：进化动作落盘前的确定性校验

ROOT = Path(__file__).parent.parent  # 项目根目录（本文件在 src/ 下）

SKILLS_DIR    = ROOT / "skills"  # 活动技能目录（进化直接改这里）
SKILLS_ORIG   = ROOT / "outputs" / "skills_original"  # 原始技能备份目录（reset 用）
VERSIONS_DIR  = ROOT / "outputs" / "skill_versions"  # 技能版本历史目录
EVAL_RUNS_DIR = ROOT / "outputs" / "eval_runs"  # 评估结果归档目录
EVAL_SET      = ROOT / "data" / "eval_set.json"  # 60 题评估集
DEMO_SCRIPT   = ROOT / "data" / "demo_script.json"  # 演示脚本（80 题分 8 块 + probe 题号）
POLICIES      = ROOT / "data" / "policies.md"  # 完整政策文档（Reviewer 专属）
BIZ_METRICS   = ROOT / "data" / "business_metrics.json"  # 商业指标声明口径（对照用）
STATE_FILE    = ROOT / "outputs" / "experiment_state.json"  # 实验状态持久化文件（跨进程）

_components: dict = {}  # 进程级组件缓存（SkillManager/Agent 等，懒加载后复用）
_state_lock = None  # 状态锁占位（下一行立即赋真值）
import threading  # 状态文件读写的线程锁
_state_lock = threading.Lock()  # 串行化 load_state/save_state，防并发写坏 JSON


# ── 实验状态持久化 ────────────────────────────────────────────────────────────

# ═════════════════════════════════════════════════════════════════
# load_state / save_state：实验状态读写（phase/当前块/各轮评估结果）
# 状态落盘 → 前端刷新或服务重启后 /state 能完整恢复现场
# ═════════════════════════════════════════════════════════════════
def load_state() -> dict:
    with _state_lock:  # 加锁防与其他线程的写操作交错
        if STATE_FILE.exists():  # 有历史状态文件
            try:
                return json.loads(STATE_FILE.read_text(encoding="utf-8"))  # 读回完整状态
            except json.JSONDecodeError:  # 文件损坏
                pass  # 静默重置（按初始状态跑）
        return {"phase": "idle", "current_block": 0, "nudge_count": 0,
                "eval_results": {}, "conv_history": []}  # 初始状态：空闲/第 0 块/无评估结果


def save_state(state: dict):
    with _state_lock:  # 加锁保证写入原子性（相对其他线程）
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)  # 确保 outputs 目录存在
        STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2),
                              encoding="utf-8")  # 序列化写盘（中文不转义）


# ═════════════════════════════════════════════════════════════════
# get_components：按进程懒加载并缓存全部实验组件
# InProcess 后端在 Web 进程内复用；Celery 后端在 Worker 进程内复用
# —— 两个进程各自持有一套，通过状态文件/事件日志对齐
# ═════════════════════════════════════════════════════════════════
def get_components() -> dict:
    """按进程懒加载实验组件（InProcess: 进程内复用；Celery: Worker 进程内复用）。"""
    global _components  # 声明修改模块级缓存
    if not _components:  # 本进程首次使用才组装（之后直接复用）
        from skill_manager import SkillManager  # Skill CRUD + 版本历史
        from evaluator import Evaluator  # 三层评估器
        from skill_store import SkillStore  # 分块索引 + 混合检索
        from agent import CustomerServiceAgent  # 客服 Agent
        from background_reviewer import BackgroundReviewer  # 失败复盘器
        from runtime_metrics import RuntimeMetrics  # 商业指标实测
        from audit import AuditLogger  # 审计留痕 + 高风险队列

        sm = SkillManager(str(SKILLS_DIR), str(VERSIONS_DIR))  # 技能管理器（skills/ + 版本目录）
        ev = Evaluator(str(EVAL_SET), mode=os.getenv("EVAL_MODE", "auto"))  # 评估器（模式可环境变量覆盖，默认自动降级）
        store = SkillStore()  # 分块索引（嵌入/向量后端按环境变量选择）
        store.reindex_all(sm.load_all())  # 启动即全量建索引（Agent 检索挂载就绪）
        agent = CustomerServiceAgent(sm, nudge_interval=0, skill_store=store)  # 主 Agent（Nudge 由流水线显式控制，关闭计数触发）
        reviewer = BackgroundReviewer(str(POLICIES), sm)  # 复盘器（持有完整政策文档）
        demo = json.loads(DEMO_SCRIPT.read_text(encoding="utf-8"))  # 读演示脚本
        qs = demo["questions"]  # 脚本中的题目序列（80 题）
        _components = {  # 组装组件表
            "sm": sm, "ev": ev, "store": store, "agent": agent,  # 核心四件套
            "reviewer": reviewer,  # 复盘器
            "metrics": RuntimeMetrics(),  # 运行时指标（自动读回历史累计）
            "audit": AuditLogger(),  # 审计器
            "probe_ids": demo.get("probe_question_ids", []),  # Probe 评估题号（回测哨兵）
            "blocks": [qs[i:i + 10] for i in range(0, len(qs), 10)],  # 按 10 题切成 8 块
        }
    return _components  # 返回（缓存的）组件表


# ═════════════════════════════════════════════════════════════════
# _judge_info：从评估器抽取最近一次判定的证据摘要
# （模式/相似度/Judge 结论与置信度），随事件透出供前端可视化
# ═════════════════════════════════════════════════════════════════
def _judge_info(ev) -> dict:
    """取最近一次判定的证据摘要（供事件透出，前端可视化）。"""
    d = getattr(ev, "last_detail", {}) or {}  # 最近一次判定的完整证据链（无则空字典）
    j = d.get("judge")  # Judge 裁决详情（可能为 None）
    info = {"mode": d.get("mode", ""), "sim": d.get("sim")}  # 基础信息：评估模式与语义相似度
    if isinstance(j, dict):  # 有 Judge 裁决
        info["judge_verdict"] = j.get("verdict")  # 结论 pass/fail
        info["judge_confidence"] = j.get("confidence")  # 置信度
    return info  # 返回摘要字典


# ═════════════════════════════════════════════════════════════════
# _retrieval_observability：用评估集测检索器命中率（hit@1/@k、MRR），
# 写入状态文件——验证"动态挂载找得到对的规则"，进化前后各测一次作对照
# ═════════════════════════════════════════════════════════════════
def _retrieval_observability(ev, store, state: dict) -> dict:
    """基于评估集的检索命中率（hit@1/hit@3/MRR），观测动态挂载的召回质量。"""
    queries = [(q["question"], q["ground_truth"].get("required", []))
               for q in sorted(ev.questions.values(), key=lambda x: x["id"])]  # 评估集全题：(问题, required 关键词) 作为预期命中词
    obs = store.hit_rate(queries, k=3)  # 跑检索命中评估（Top-3）
    state["retrieval_obs"] = {k: obs[k] for k in ("total", "hit_at_1", "hit_at_k", "mrr")}  # 摘要存入状态（/state 透出）
    save_state(state)  # 落盘
    return obs  # 返回完整观测结果


def _load_business_metrics() -> dict:
    if BIZ_METRICS.exists():  # 声明口径文件存在
        return json.loads(BIZ_METRICS.read_text(encoding="utf-8"))  # 读回指标列表
    return {"metrics": []}  # 缺失则空结构（终局事件里少一节）


# ── 任务分发 ──────────────────────────────────────────────────────────────────

# ═════════════════════════════════════════════════════════════════
# run_task：后台 Worker 唯一入口——按任务名分发到三个纯函数
# baseline（基线）→ block N（块进化，8 次）→ final（终局评估）
# ═════════════════════════════════════════════════════════════════
def run_task(task: str, params: dict, emit) -> dict:
    """后台 Worker 入口。task ∈ {baseline, block, final}。"""
    if task == "baseline":  # 基线评估
        return run_baseline(emit)  # 跑 60 题得到初始准确率（约 33%）
    if task == "block":  # 块进化
        return run_block(params.get("block_index", 0), emit)  # 跑第 N 块 10 题 + 失败复盘 + 回测回滚
    if task == "final":  # 最终评估
        return run_final(emit)  # 再跑 60 题得到终局准确率（约 93%）
    raise ValueError(f"未知任务: {task}")  # 未知任务直接报错


# ── 基线评估 ──────────────────────────────────────────────────────────────────

# ═════════════════════════════════════════════════════════════════
# _answer_and_record：单题全链路——计时回答 → 三层判分 →
# 运行时指标累计 + 审计留痕。三个 run_* 与 probe 共用的记账中枢
# ═════════════════════════════════════════════════════════════════
def _answer_and_record(agent, ev, question: str, qid: int, run_id: str,
                       metrics, audit):
    """回答 + 判定 + 运行时指标/审计留痕，返回 (answer, ok, reason, latency_ms)。"""
    t0 = time.perf_counter()  # 高精度计时起点
    answer = agent.answer(question)  # Agent 用当前技能生成回答（含检索挂载）
    latency_ms = (time.perf_counter() - t0) * 1000  # 端到端延迟（毫秒）
    ok, reason = ev.evaluate_answer(answer, qid)  # 三层评估判分
    metrics.record_answer(question=question, answer=answer, correct=ok,
                          deferred=reason.startswith("Agent 推脱"),  # 推脱判定以固定前缀识别
                          latency_ms=latency_ms, run_id=run_id)  # 累计分流率/延迟指标
    audit.log_answer(question=question, answer=answer, correct=ok, reason=reason,
                     run_id=run_id, mount=getattr(agent, "last_mount", {}),  # 挂载信息（哪些分块进了 Prompt）
                     judge=_judge_info(ev), latency_ms=latency_ms)  # Judge 证据与延迟入审计
    return answer, ok, reason, latency_ms  # 返回四元组给调用方


# ═════════════════════════════════════════════════════════════════
# run_baseline：基线评估——60 题全量跑一遍，产出初始准确率锚点
# 同时保存初始技能版本快照、检索命中率基线，供后续进化对照
# ═════════════════════════════════════════════════════════════════
def run_baseline(emit) -> dict:
    c = get_components()  # 取组件（懒加载）
    sm, ev, agent, store = c["sm"], c["ev"], c["agent"], c["store"]  # 解包核心组件
    metrics, audit = c["metrics"], c["audit"]  # 指标与审计
    state = load_state()  # 读实验状态
    state.update(phase="baseline_running", eval_results={})  # 进入基线阶段并清空旧评估结果（新一轮实验）
    save_state(state)  # 落盘状态

    # 保存初始快照
    for name, content in sm.load_all().items():  # 遍历当前全部技能
        sm._save_version(name, content, action="initial", reason="初始版本")  # 每个种子技能记一条 v-initial 历史

    # 检索可观测性：初始索引的 hit@1/hit@3/MRR
    obs = _retrieval_observability(ev, store, state)  # 测初始检索命中率
    emit("retrieval_hitrate", **{k: obs[k] for k in
        ("total", "hit_at_1", "hit_at_k", "mrr")}, stage="baseline")  # 事件透出（标 baseline 阶段）

    all_ids = sorted(ev.questions.keys())  # 全部题号（有序）
    by_cat: dict = {}  # 分类统计
    correct = 0  # 答对计数
    emit("phase_change", phase="baseline_running")  # 通知前端进入基线阶段

    for qid in all_ids:  # 逐题评估（60 题）
        q = ev.questions[qid]  # 取题
        answer, ok, reason, latency_ms = _answer_and_record(
            agent, ev, q["question"], qid, "baseline", metrics, audit)  # 单题全链路（含记账）
        correct += int(ok)  # 累计对数
        cat = q["category"]  # 类目
        by_cat.setdefault(cat, {"total": 0, "correct": 0})  # 类目桶初始化
        by_cat[cat]["total"] += 1  # 类目总数 +1
        by_cat[cat]["correct"] += int(ok)  # 类目对数 +1
        emit("eval_q", run_id="baseline", id=qid, correct=ok, category=cat,  # 逐题事件（前端对话流渲染）
             question=q["question"][:60], answer=answer[:200],  # 问题/回答摘要
             fail_reason="" if ok else reason,  # 失败原因（对则空）
             latency_ms=round(latency_ms, 1),  # 延迟
             mount=agent.last_mount, judge=_judge_info(ev))  # 挂载信息 + 判定证据

    for v in by_cat.values():  # 计算各类目准确率
        v["accuracy"] = round(v["correct"] / v["total"], 3)  # 3 位小数

    result = {"run_id": "baseline", "accuracy": round(correct / len(all_ids), 3),  # 基线结果：总准确率
              "correct": correct, "total": len(all_ids), "by_category": by_cat,  # 对错数与分类明细
              "timestamp": datetime.now().isoformat(),  # 时间戳
              "skill_versions_active": sm.get_active_versions()}  # 当时的技能版本组合（溯源）
    state["eval_results"]["baseline"] = result  # 结果存入状态（Reverse Eval Gate 的兜底锚点）
    state["conv_history"] = []  # 清空对话历史（基线样本不进复盘）
    state["current_block"] = 0  # 块指针复位（下一步跑第 0 块）
    state["phase"] = "baseline_done"  # 阶段推进
    save_state(state)  # 落盘
    metrics.save()  # 指标落盘

    emit("eval_complete", run_id="baseline", correct=correct, total=len(all_ids),  # 基线完成事件
         accuracy=result["accuracy"], by_category=by_cat,  # 汇总指标
         judge_stats=dict(ev.stats), runtime_metrics=metrics.snapshot())  # Judge 统计 + 实测指标快照
    emit("phase_change", phase="baseline_done", current_block=0)  # 阶段变更事件
    return result  # 返回基线结果给 Worker（写入任务返回值）


# ── 块进化 ────────────────────────────────────────────────────────────────────

# ═════════════════════════════════════════════════════════════════
# run_block：单块进化（闭环核心）——10 题问答 → 失败样本收集 →
# Reviewer 生成动作 → 红线校验 → 落盘 + 重建索引 → Probe 回测 →
# Reverse Eval Gate 劣化则回滚。每块一次完整的"评估→复盘→进化→回测"
# ═════════════════════════════════════════════════════════════════
def run_block(block_index: int, emit) -> dict:
    c = get_components()  # 取组件
    sm, ev, agent, reviewer, store = (c["sm"], c["ev"], c["agent"],
                                      c["reviewer"], c["store"])  # 解包
    metrics, audit = c["metrics"], c["audit"]  # 指标与审计
    state = load_state()  # 读状态
    if block_index != state["current_block"]:  # 块序号与状态指针不一致（跳块/重复提交）
        raise RuntimeError(f"需要运行第 {state['current_block']} 块，收到 {block_index}")  # 拒绝执行（保证顺序）

    block_qs = c["blocks"][block_index]  # 取该块 10 题
    block_name = block_qs[0].get("block", f"block_{block_index}")  # 块名（脚本里定义，如"开机与显示"）
    state["phase"] = f"block_{block_index}_running"  # 更新阶段
    save_state(state)  # 落盘
    emit("phase_change", phase=state["phase"], current_block=block_index)  # 通知前端

    # Phase 1: 逐题回答 + 累积失败样本
    block_correct = 0  # 本块对数
    block_failed_turns: list[dict] = []  # 本块失败样本（复盘输入）
    for item in block_qs:  # 遍历本块题目
        q = ev.questions.get(item["eval_id"], {})  # 关联评估集信息（类目等；可能无）
        emit("question_start", q_seq=item["seq"], question=item["question"],  # 开题事件（前端渲染用户气泡）
             block=block_name, eval_id=item["eval_id"])  # 块名与关联评估题号
        answer, ok, reason, latency_ms = _answer_and_record(
            agent, ev, item["question"], item["eval_id"],
            f"block_{block_index}", metrics, audit)  # 单题全链路（回答+判分+记账）
        block_correct += int(ok)  # 累计对数
        if not ok:  # 答错/推脱
            block_failed_turns.append(
                {"question": item["question"], "answer": answer, "fail_reason": reason})  # 收集失败样本（问题/回答/判错理由）
        state["conv_history"] = (state.get("conv_history", [])
                                 + [{"question": item["question"], "answer": answer}])[-30:]  # 滚动保留最近 30 条对话
        save_state(state)  # 逐题落盘（断线可恢复到题级）
        emit("question_result", q_seq=item["seq"], answer=answer, correct=ok,  # 答题结果事件（渲染客服气泡+对错标记）
             fail_reason="" if ok else reason, category=q.get("category", ""),  # 失败原因与类目
             latency_ms=round(latency_ms, 1),  # 延迟
             mount=agent.last_mount, judge=_judge_info(ev))  # 挂载与判定证据

    block_acc = block_correct / len(block_qs)  # 本块准确率
    emit("block_complete", block=block_name, correct=block_correct,
         total=len(block_qs), accuracy=round(block_acc, 3))  # 块完成事件

    executed = []  # 本块实际落盘成功的进化动作
    if not block_failed_turns:  # 本块全对
        emit("nudge_skipped", block=block_name,
             reason="本块全部答对，跳过 Nudge 和 Probe eval")  # 无需复盘，跳过后续
    else:  # 有失败样本 → 触发 Nudge 复盘
        state["nudge_count"] = state.get("nudge_count", 0) + 1  # Nudge 计数 +1
        save_state(state)  # 落盘
        emit("nudge_start", nudge_num=state["nudge_count"], block=block_name,
             failure_count=len(block_failed_turns))  # Nudge 开始事件（含失败数）

        actions = reviewer.review(block_failed_turns)  # 复盘器分析失败样本 → 动作列表（LLM 驱动）
        emit("reviewer_analysis", analysis=getattr(reviewer, "last_analysis", ""))  # 透出分析文本（进化日志展示）

        for act in (actions or []):  # 逐个执行动作（可能 0~N 个）
            try:  # 单个动作失败不影响其余动作
                # 安全红线守卫：LLM 生成的规则与医疗红线冲突 → 阻断，不落盘
                violations = redline_check_action(act)  # 红线扫描（确定性正则）
                if violations:  # 命中红线
                    emit("redline_blocked", action=act.get("action"),
                         skill_name=act.get("skill_name"),
                         violations=[v["message"] for v in violations])  # 事件透出阻断详情
                    continue  # 跳过该动作（不落盘）
                if act["action"] == "create":  # 新建技能
                    ok_act = sm.create(act["skill_name"], act["content"],
                                       reason=act.get("reason", ""))  # 落盘新 SKILL.md + v1 历史
                elif act["action"] == "patch":  # 修补技能
                    ok_act = sm.patch(act["skill_name"], act["old_text"],
                                      act["new_text"], reason=act.get("reason", ""))  # 精确替换 + 版本 +1
                else:  # 未知动作类型
                    ok_act = False  # 视为失败
                if ok_act:  # 动作执行成功
                    executed.append({"action": act["action"],
                                     "skill_name": act["skill_name"],
                                     "reason": act.get("reason", "")[:120]})  # 记入已执行清单（回滚依据）
                    if act["action"] == "create":  # 新技能上线
                        metrics.record_skill_created(act["skill_name"],
                                                     f"block_{block_index}")  # 记录上线事件（类目周期指标）
                    emit("skill_action", **executed[-1])  # 事件透出（进化日志渲染）
            except Exception as e:  # 动作执行异常
                emit("skill_error", error=str(e)[:100])  # 事件透出错误

        emit("nudge_complete", num_actions=len(executed))  # Nudge 完成事件（含动作数）

        # Skill 变更 → 增量重建向量索引（动作开关：这里全量重建，毫秒级）
        if executed:  # 本块有成功落盘的变更
            store.reindex_all(sm.load_all())  # 全量重建分块索引（新规则即时可检索）
            emit("retrieval_info", **store.stats(),
                 reason="Skill 变更后重建索引")  # 事件透出索引信息
            obs = _retrieval_observability(ev, store, state)  # 复测检索命中率
            emit("retrieval_hitrate", **{k: obs[k] for k in
                ("total", "hit_at_1", "hit_at_k", "mrr")},
                stage=f"after_block_{block_index}")  # 事件透出（与基线对照）

        # Probe 评估（仅触发 Nudge 时）
        emit("probe_start", total=len(c["probe_ids"]))  # Probe 开始事件（哨兵题回测）
        result = _run_probe(ev, agent, sm, c["probe_ids"],
                            f"after_block_{block_index}", metrics, audit)  # 跑 Probe 题集（含记账）
        emit("probe_result", run_id=result["run_id"], correct=result["correct"],
             total=result["total"], accuracy=result["accuracy"],
             by_category=result["by_category"],
             skill_versions=result["skill_versions_active"])  # 事件透出 Probe 结果

        # Reverse Eval Gate：进化后整体准确率显著劣化 → 自动回滚本轮 Skill 变更
        if executed:  # 只有真的改了技能才需要闸门
            gate = _reverse_eval_gate(state, result["accuracy"], block_index)  # 与锚点对比判定
            if not gate["pass"]:  # 判定为有害进化
                emit("rollback_reverted", reason=gate["reason"],
                     anchor=gate["anchor"], probe_acc=result["accuracy"])  # 事件透出回滚原因
                for act in reversed(executed):  # 逆序回滚（与执行顺序相反，版本栈正确弹出）
                    try:
                        sm.rollback(act["skill_name"], reason="reverse_eval_gate")  # 回滚单个技能
                    except Exception as e:  # 回滚失败
                        emit("skill_error", error=f"rollback {act['skill_name']}: {e}"[:100])  # 事件透出
                store.reindex_all(sm.load_all())  # 回滚后重建索引（恢复旧知识）
                # 回滚后复测 Probe，确认闸门恢复
                result2 = _run_probe(ev, agent, sm, c["probe_ids"],
                                     f"after_block_{block_index}_rollback",
                                     metrics, audit)  # 复测（run_id 带 _rollback 后缀）
                emit("probe_result", run_id=result2["run_id"],
                     correct=result2["correct"], total=result2["total"],
                     accuracy=result2["accuracy"], by_category=result2["by_category"],
                     skill_versions=result2["skill_versions_active"])  # 事件透出复测结果

    # 推进状态
    state["current_block"] = block_index + 1  # 块指针 +1（下一次跑下一块）
    is_last = state["current_block"] >= len(c["blocks"])  # 是否已跑完全部块
    state["phase"] = "all_blocks_done" if is_last else f"block_{block_index}_done"  # 阶段推进
    save_state(state)  # 落盘
    metrics.save()  # 指标落盘
    emit("phase_change", phase=state["phase"], current_block=state["current_block"])  # 阶段变更事件
    return {"block": block_name, "correct": block_correct,
            "total": len(block_qs), "actions": len(executed)}  # 返回块执行摘要


# ═════════════════════════════════════════════════════════════════
# _reverse_eval_gate：反向评估闸门——Probe 准确率与历史锚点对比，
# 劣化超阈值（默认 10 个百分点）即判"有害进化"，触发整轮回滚
# ═════════════════════════════════════════════════════════════════
def _reverse_eval_gate(state: dict, probe_accuracy: float, block_index: int) -> dict:
    """Reverse Eval Gate：进化后 Probe 准确率与锚点对比，显著劣化则建议回滚。

    锚点选取：最近一次历史 Probe 准确率；无历史则用基线评估准确率。
    阈值：低于锚点 10 个百分点即触发回滚（RETRIEVE_GATE_MARGIN 可调）。"""
    margin = float(os.getenv("EVAL_GATE_MARGIN", "0.10"))  # 劣化容忍度（默认 0.10）
    anchor = None  # 锚点准确率
    for run_id, res in (state.get("eval_results") or {}).items():  # 遍历历史评估结果
        if run_id.startswith("after_block") and run_id.endswith(str(block_index)):  # 本轮自身的 Probe（含 _rollback 复测）
            continue                           # 跳过本轮自身
        if run_id.startswith("after_block"):  # 其他块的历史 Probe
            acc = res.get("accuracy")  # 取准确率
            if acc is not None and (anchor is None or acc > anchor):
                anchor = acc                    # 取最近（更晚块）的准确率
    if anchor is None:  # 没有任何历史 Probe
        anchor = (state.get("eval_results") or {}).get("baseline", {}).get("accuracy")  # 退回用基线准确率当锚点
    if anchor is None:  # 连基线都没有（异常路径）
        return {"pass": True, "anchor": None,
                "reason": "无可用锚点，跳过反向闸门（首批进化不做回滚）"}  # 放行（无从对比）
    if probe_accuracy < anchor - margin:  # 劣化超过容忍度
        return {"pass": False, "anchor": round(anchor, 3),
                "reason": f"Probe {probe_accuracy:.1%} 低于锚点 {anchor:.1%} "
                          f"超过 {margin:.0%}，判定为有害进化"}  # 拦截 → 调用方执行回滚
    return {"pass": True, "anchor": round(anchor, 3),
            "reason": f"Probe {probe_accuracy:.1%} ≥ 锚点 {anchor:.1%}-{margin:.0%}，进化保留"}  # 通过 → 保留进化


# ═════════════════════════════════════════════════════════════════
# _run_probe：哨兵题回测——用固定 probe 题集快速验证进化效果
# 结果写入状态（供闸门取锚点），带记账（metrics/audit 可选）
# ═════════════════════════════════════════════════════════════════
def _run_probe(ev, agent, sm, probe_ids: list[int], run_id: str,
               metrics=None, audit=None) -> dict:
    by_cat: dict = {}  # 分类统计
    correct = 0  # 对数
    for qid in probe_ids:  # 遍历哨兵题
        q = ev.questions[qid]  # 取题
        if metrics is not None and audit is not None:  # 带记账模式（正式回测）
            answer, ok, _, _ = _answer_and_record(agent, ev, q["question"], qid,
                                                  run_id, metrics, audit)  # 全链路（指标/审计同步累计）
        else:  # 轻量模式（不记账）
            answer = agent.answer(q["question"])  # 只回答
            ok, _ = ev.evaluate_answer(answer, qid)  # 只判分
        correct += int(ok)  # 累计对数
        by_cat.setdefault(q["category"], {"total": 0, "correct": 0})  # 类目桶
        by_cat[q["category"]]["total"] += 1  # 总数 +1
        by_cat[q["category"]]["correct"] += int(ok)  # 对数 +1
    for v in by_cat.values():  # 类目准确率
        v["accuracy"] = round(v["correct"] / v["total"], 3)  # 3 位小数
    result = {"run_id": run_id, "accuracy": round(correct / len(probe_ids), 3),  # Probe 结果
              "correct": correct, "total": len(probe_ids), "by_category": by_cat,
              "timestamp": datetime.now().isoformat(),  # 时间戳
              "skill_versions_active": sm.get_active_versions()}  # 版本快照（溯源）
    state = load_state()  # 读状态
    state["eval_results"][run_id] = result  # 存入（闸门下次取它当锚点）
    save_state(state)  # 落盘
    return result  # 返回结果


# ── 最终评估 ──────────────────────────────────────────────────────────────────

# ═════════════════════════════════════════════════════════════════
# run_final：终局评估——8 块进化结束后再跑全量 60 题，
# 与基线对照得到进化增益（33%→93%），并写入商业指标/审计汇总
# ═════════════════════════════════════════════════════════════════
def run_final(emit) -> dict:
    c = get_components()  # 取组件
    sm, ev, agent = c["sm"], c["ev"], c["agent"]  # 解包
    metrics, audit = c["metrics"], c["audit"]  # 指标与审计
    state = load_state()  # 读状态
    state["phase"] = "final_running"  # 进入终局阶段
    save_state(state)  # 落盘
    emit("phase_change", phase="final_running")  # 通知前端

    all_ids = sorted(ev.questions.keys())  # 全部题号
    by_cat: dict = {}  # 分类统计
    correct = 0  # 对数
    for qid in all_ids:  # 逐题评估（60 题）
        q = ev.questions[qid]  # 取题
        answer, ok, reason, latency_ms = _answer_and_record(
            agent, ev, q["question"], qid, "final", metrics, audit)  # 单题全链路
        correct += int(ok)  # 累计
        cat = q["category"]  # 类目
        by_cat.setdefault(cat, {"total": 0, "correct": 0})  # 桶初始化
        by_cat[cat]["total"] += 1  # 总数 +1
        by_cat[cat]["correct"] += int(ok)  # 对数 +1
        emit("eval_q", run_id="final", id=qid, correct=ok, category=cat,  # 逐题事件（前端渲染）
             question=q["question"][:60], answer=answer[:200],
             fail_reason="" if ok else reason,
             latency_ms=round(latency_ms, 1),
             mount=agent.last_mount, judge=_judge_info(ev))  # 挂载与证据

    for v in by_cat.values():  # 类目准确率
        v["accuracy"] = round(v["correct"] / v["total"], 3)  # 3 位小数

    result = {"run_id": "final", "accuracy": round(correct / len(all_ids), 3),  # 终局结果
              "correct": correct, "total": len(all_ids), "by_category": by_cat,
              "timestamp": datetime.now().isoformat(),
              "skill_versions_active": sm.get_active_versions()}  # 终局版本快照
    state["eval_results"]["final"] = result  # 存入状态
    state["phase"] = "complete"  # 实验完成
    save_state(state)  # 落盘

    metrics.record_evolution(
        (state["eval_results"].get("baseline") or {}).get("accuracy"),
        result["accuracy"])  # 记录基线→终局准确率对照（进化增益指标）
    metrics.save()  # 指标落盘

    emit("eval_complete", run_id="final", correct=correct, total=len(all_ids),  # 终局完成事件（前端渲染总览）
         accuracy=result["accuracy"], by_category=by_cat,
         judge_stats=dict(ev.stats), business_metrics=_load_business_metrics(),  # Judge 统计 + 声明口径商业指标
         runtime_metrics=metrics.snapshot(), audit_stats=audit.stats())  # 实测指标 + 审计汇总
    emit("phase_change", phase="complete")  # 阶段完成事件
    return result  # 返回终局结果
