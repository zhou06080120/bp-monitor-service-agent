"""
演示主程序：运行 Skill 自进化实验，输出多版本对比数据。

教学重点：
  完整展示"无进化基线 → 运行时 Nudge 多轮进化"的量化效果对比。
  每次 Nudge 后跑 30 题 probe eval，记录各版本对同一问题的回答差异。

使用方式：
  cd bp_monitor_service_agent
  python src/demo_runner.py

依赖：
  pip install openai
  set DEEPSEEK_API_KEY=your_key
"""

import os  # 检查 DEEPSEEK_API_KEY 环境变量
import sys  # 注入 src 路径 + 无 Key 时退出
import json  # 读取演示脚本/商业指标，写评估结果与进化日志
import shutil  # 技能目录的备份（copytree）与还原（rmtree+copytree）
from pathlib import Path  # 统一管理各目录路径
from datetime import datetime  # 评估结果与事件的时间戳

ROOT = Path(__file__).parent.parent  # 项目根目录（本文件在 src/ 下）
sys.path.insert(0, str(ROOT / "src"))  # 让 src 内模块可按顶级名导入

from skill_manager import SkillManager  # Skill CRUD + 版本历史 + 回滚
from evaluator import Evaluator  # 三层评估器（规则→语义→Judge）
from agent import CustomerServiceAgent  # 客服 Agent（检索挂载 + 回答）
from background_reviewer import BackgroundReviewer  # 失败复盘器（LLM 生成进化动作）
from skill_store import SkillStore  # 分块索引 + 混合检索
from redline_guard import check_action as redline_check_action  # 红线守卫（进化动作校验）

SKILLS_DIR    = ROOT / "skills"  # 活动技能目录
SKILLS_ORIG   = ROOT / "outputs" / "skills_original"   # 只建一次，永不覆盖
EVAL_SET      = ROOT / "data" / "eval_set.json"  # 60 题评估集
DEMO_SCRIPT   = ROOT / "data" / "demo_script.json"  # 演示脚本（题目序列+Nudge 间隔+probe 题号）
POLICIES      = ROOT / "data" / "policies.md"  # 完整政策（Reviewer 专属）
VERSIONS_DIR  = ROOT / "outputs" / "skill_versions"  # 版本历史目录
EVAL_RUNS_DIR = ROOT / "outputs" / "eval_runs"  # 每次评估结果一个 JSON
EVOL_LOG      = ROOT / "outputs" / "evolution_log.json"  # 实验总日志（跨版本对比数据）


# ── 备份 / 还原 ──────────────────────────────────────────────────────────────

# ═════════════════════════════════════════════════════════════════
# ensure_original：首次运行把 skills/ 整树备份到 outputs/skills_original
# 并为每个种子技能落 v1 快照；之后只读不写——保证实验永远可重跑
# ═════════════════════════════════════════════════════════════════
def ensure_original(sm: SkillManager):
    """首次运行时创建原始备份（并保存快照 v1），之后只读不写。"""
    if not SKILLS_ORIG.exists():  # 备份尚不存在（首次运行）
        shutil.copytree(SKILLS_DIR, SKILLS_ORIG)  # 整树复制技能目录
        print(f"✓ 首次运行：原始 Skills 备份至 {SKILLS_ORIG.name}/")  # 提示备份位置
        # 把初始 Skills 也存入版本历史（v1 快照）
        for skill_name, content in sm.load_all().items():  # 遍历每个种子技能
            sm._save_version(skill_name, content, action="initial", reason="初始版本")  # 落 initial 历史（版本 1）
    else:  # 已有备份
        print("✓ 检测到原始备份，已跳过覆盖")  # 绝不覆盖（保持原始种子）

# ═════════════════════════════════════════════════════════════════
# restore_from_original：每次实验前把 skills/ 还原成原始种子，
# 并清空版本历史/快照/评估归档——保证多次运行结果可复现对比
# ═════════════════════════════════════════════════════════════════
def restore_from_original():
    """每次实验前从原始备份还原，保证可重复运行。"""
    if not SKILLS_ORIG.exists():  # 备份不存在（异常状态）
        raise RuntimeError("原始备份不存在，请删除 outputs/ 并重新运行")  # 指引修复
    if SKILLS_DIR.exists():  # 现有技能目录
        shutil.rmtree(SKILLS_DIR)  # 整树删除（含进化产物）
    shutil.copytree(SKILLS_ORIG, SKILLS_DIR)  # 从原始备份复制回干净的种子
    # 清空版本历史和快照，重新开始
    if VERSIONS_DIR.exists():  # 历史目录存在
        shutil.rmtree(VERSIONS_DIR)  # 删除（全部重来）
    snapshots = ROOT / "outputs" / "skill_snapshots"  # 快照目录
    if snapshots.exists():  # 存在
        shutil.rmtree(snapshots)  # 删除
    if EVAL_RUNS_DIR.exists():  # 评估归档存在
        shutil.rmtree(EVAL_RUNS_DIR)  # 删除
    print("✓ 已还原初始 Skills，清空上次版本历史")  # 提示还原完成


# ── Probe / Full Eval ─────────────────────────────────────────────────────────

# ═════════════════════════════════════════════════════════════════
# run_probe_eval：对指定题集（probe 哨兵题或全量 60 题）跑评估，
# 结果归档到 eval_runs/{run_id}.json；per-question 答案留作跨版本对比
# ═════════════════════════════════════════════════════════════════
def run_probe_eval(agent: CustomerServiceAgent, evaluator: Evaluator,
                   probe_ids: list[int], run_id: str, label: str,
                   sm: SkillManager) -> dict:
    """
    对 probe_ids 指定的题目运行评估，保存到 eval_runs/{run_id}.json。
    返回包含 per-question 答案的完整结果字典。
    """
    EVAL_RUNS_DIR.mkdir(parents=True, exist_ok=True)  # 确保归档目录存在
    total, correct = 0, 0  # 总数与对数
    by_category: dict = {}  # 分类统计
    answers: dict = {}  # 逐题答案（qid → {answer, correct, fail_reason}）

    for qid in probe_ids:  # 遍历本次评估的题目
        q = evaluator.questions[qid]  # 取题
        answer = agent.answer(q["question"])  # Agent 用当前技能回答
        ok, reason = evaluator.evaluate_answer(answer, qid)  # 三层判分
        total += 1  # 总数 +1
        cat = q["category"]  # 类目
        by_category.setdefault(cat, {"total": 0, "correct": 0})  # 桶初始化
        by_category[cat]["total"] += 1  # 类目总数 +1
        if ok:  # 答对
            correct += 1  # 对数 +1
            by_category[cat]["correct"] += 1  # 类目对数 +1
        answers[str(qid)] = {"answer": answer, "correct": ok, "fail_reason": reason if not ok else ""}  # 记录该题答案（键转字符串便于 JSON）

    for cat in by_category.values():  # 类目准确率
        cat["accuracy"] = round(cat["correct"] / cat["total"], 3)  # 3 位小数

    result = {  # 组装完整结果
        "run_id": run_id,  # 运行标识（baseline / after_nudge_seqN / final）
        "label": label,  # 人类可读标签
        "timestamp": datetime.now().isoformat(),  # 时间戳
        "skill_versions_active": sm.get_active_versions(),  # 本轮生效的技能版本（溯源）
        "summary": {"total": total, "correct": correct, "accuracy": round(correct / total, 3)},  # 汇总
        "by_category": by_category,  # 分类明细
        "answers": answers,  # 逐题答案（对比用）
    }
    (EVAL_RUNS_DIR / f"{run_id}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )  # 归档落盘（中文不转义、缩进 2）
    return result  # 返回完整结果

# ═════════════════════════════════════════════════════════════════
# run_full_eval：全量评估 = 对评估集全部题目跑 run_probe_eval
# ═════════════════════════════════════════════════════════════════
def run_full_eval(agent: CustomerServiceAgent, evaluator: Evaluator,
                  run_id: str, label: str, sm: SkillManager) -> dict:
    all_ids = list(evaluator.questions.keys())  # 评估集全部题号
    return run_probe_eval(agent, evaluator, all_ids, run_id, label, sm)  # 复用 probe 逻辑跑全量


# ── Evolution Log ─────────────────────────────────────────────────────────────

# ═════════════════════════════════════════════════════════════════
# EvolutionLog：实验全程的结构化数据收集器
# 三类数据：各轮评估摘要 / Nudge 事件 / 每题跨版本答案对比，
# 实验结束后统一写入 evolution_log.json 供复盘分析
# ═════════════════════════════════════════════════════════════════
class EvolutionLog:
    """收集实验过程中所有结构化数据，实验结束后写入 evolution_log.json。"""

    def __init__(self):
        self.eval_runs: list[dict] = []  # 各轮评估摘要列表
        self.nudge_events: list[dict] = []  # 各次 Nudge 事件列表
        self.question_history: dict[str, list] = {}   # qid -> [{run_id, answer, correct}]

    def add_eval_run(self, result: dict):
        self.eval_runs.append({  # 追加一轮评估摘要
            "run_id": result["run_id"],  # 运行标识
            "label": result["label"],  # 标签
            "timestamp": result["timestamp"],  # 时间
            "skill_versions_active": result["skill_versions_active"],  # 版本组合
            "accuracy": result["summary"]["accuracy"],  # 准确率
            "correct": result["summary"]["correct"],  # 对数
            "total": result["summary"]["total"],  # 总数
            "by_category": {k: v["accuracy"] for k, v in result["by_category"].items()},  # 分类准确率压缩
        })
        # 更新 per-question 历史
        for qid_str, ans_data in result["answers"].items():  # 遍历本轮逐题答案
            self.question_history.setdefault(qid_str, []).append({  # 该题追加一条版本记录
                "run_id": result["run_id"],  # 哪一轮
                "label": result["label"],  # 标签
                "skill_versions": result["skill_versions_active"],  # 当轮版本组合
                "answer": ans_data["answer"],  # 当轮答案
                "correct": ans_data["correct"],  # 对错
                "fail_reason": ans_data.get("fail_reason", ""),  # 失败原因
            })

    def add_nudge_event(self, seq: int, block: str, actions_taken: list[dict],
                        accuracy_before: float, skill_versions_after: dict):
        self.nudge_events.append({  # 追加一次 Nudge 事件
            "after_seq": seq,  # 触发时的题序号
            "block": block,  # 所在块名
            "timestamp": datetime.now().isoformat(),  # 时间
            "accuracy_before_this_block": round(accuracy_before, 3),  # 本块（进化前）准确率
            "actions_taken": actions_taken,  # 实际执行的动作清单
            "skill_versions_after": skill_versions_after,  # 进化后的版本组合
        })

    def save(self, sm: SkillManager, evaluator: Evaluator):
        """写入 evolution_log.json，包含 Skill 版本历史和题目对比数据。"""
        skill_snapshots = {}  # 技能名 → 版本历史摘要
        for skill_dir in SKILLS_DIR.iterdir():  # 遍历技能目录
            if skill_dir.is_dir():  # 只处理目录
                name = skill_dir.name  # 技能名
                history = sm.get_version_history(name)  # 读完整历史
                skill_snapshots[name] = [  # 压缩为摘要列表
                    {
                        "version": h["version"],  # 版本号
                        "time": h["time"],  # 时间
                        "action": h["action"],  # 动作类型
                        "reason": h["reason"][:120],  # 原因（截断）
                        "snapshot_file": h.get("snapshot_file", ""),  # 快照文件路径
                    }
                    for h in history  # 遍历每条历史
                ]

        # 构建 question_comparison：所有题目的完整跨版本答案对比
        question_comparison = {}  # qid → 题目信息 + 历次答案
        for qid_str, history in self.question_history.items():  # 遍历收集到的每题历史
            qid = int(qid_str)  # 转回整数题号
            q = evaluator.questions.get(qid)  # 关联评估集信息
            if q:  # 题目存在
                question_comparison[qid_str] = {
                    "question": q["question"],  # 问题原文
                    "category": q["category"],  # 类目
                    "difficulty": q["difficulty"],  # 难度
                    "ground_truth": q["ground_truth"],  # 判分要点
                    "history": history,  # 历轮答案记录
                }

        log = {  # 组装总日志
            "generated_at": datetime.now().isoformat(),  # 生成时间
            "skill_snapshots": skill_snapshots,  # 技能版本历史
            "eval_runs": self.eval_runs,  # 各轮评估
            "nudge_events": self.nudge_events,  # Nudge 事件
            "question_comparison": question_comparison,  # 逐题跨版本对比
        }
        EVOL_LOG.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")  # 写盘
        print(f"✓ evolution_log.json 已保存 ({EVOL_LOG})")  # 提示保存位置


# ── 主实验流程 ────────────────────────────────────────────────────────────────

# ═════════════════════════════════════════════════════════════════
# run_experiment：命令行一键演示主流程（与 Web 版 pipeline 同构）
# 备份还原 → 基线 60 题 → 8 块×10 题（失败则 Nudge 复盘+红线+落盘
# +Probe 回测+反向闸门回滚）→ 终局 60 题 → 汇总打印 + 落 evolution_log
# ═════════════════════════════════════════════════════════════════
def run_experiment():
    print("=" * 60)  # 分隔线
    print("  康康血压计客服 Agent 自进化实验 v2")  # 实验标题
    print("=" * 60)  # 分隔线

    # 准备目录
    (ROOT / "outputs").mkdir(exist_ok=True)  # 确保 outputs 目录存在
    sm = SkillManager(str(SKILLS_DIR), str(VERSIONS_DIR))  # 技能管理器

    # 第一步：确保原始备份存在，然后还原
    ensure_original(sm)  # 首次运行建备份（含 v1 快照）
    restore_from_original()  # 还原种子技能 + 清空历史

    # 还原后需要重新初始化（路径没变，但文件被替换了）
    sm = SkillManager(str(SKILLS_DIR), str(VERSIONS_DIR))  # 重建管理器
    agent = CustomerServiceAgent(sm, nudge_interval=0)  # nudge 由 demo_runner 手动触发
    reviewer = BackgroundReviewer(str(POLICIES), sm)  # 复盘器（持完整政策）
    evaluator = Evaluator(str(EVAL_SET))  # 评估器（auto 模式）
    store = SkillStore()  # 分块索引
    store.reindex_all(sm.load_all())  # 全量建索引
    agent.skill_store = store     # 向量检索 + 动态按需挂载
    elog = EvolutionLog()  # 进化日志收集器
    print(f"  评估模式: {evaluator.mode} | Skill 索引: {store.stats()}")  # 打印运行形态
    _queries = [(q["question"], q["ground_truth"].get("required", []))
                for q in evaluator.questions.values()]  # 构造检索命中率评估的 (问题, 关键词) 对
    _obs = store.hit_rate(_queries, k=3)  # 初始检索命中率
    print(f"  检索命中率: hit@1={_obs['hit_at_1']:.0%}  hit@3={_obs['hit_at_k']:.0%}  MRR={_obs['mrr']}")  # 打印检索基线

    # 保存初始 Skill 快照（v1）
    for skill_name, content in sm.load_all().items():  # 遍历种子技能
        sm._save_version(skill_name, content, action="initial", reason="初始版本")  # 落 v1 历史

    # 加载脚本
    script_data = json.loads(DEMO_SCRIPT.read_text(encoding="utf-8"))  # 读演示脚本
    demo_questions = script_data["questions"]  # 80 题序列
    nudge_interval = script_data["nudge_interval"]  # Nudge 间隔（每 10 题触发）
    probe_ids = script_data.get("probe_question_ids", list(range(1, 31)))  # probe 题号（默认前 30 题）

    # ── 基线评估 ─────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")  # 分节线
    print("基线评估（初始 Skills，无进化）")  # 分节标题
    print("─" * 60)  # 分节线
    baseline = run_full_eval(agent, evaluator, "baseline", "基线（初始Skills）", sm)  # 跑 60 题基线（约 33%）
    elog.add_eval_run(baseline)  # 记入日志
    _print_summary(baseline)  # 打印摘要
    agent.conversation_history.clear()  # 清空对话历史（基线样本不进复盘）

    # ── 演示脚本 ─────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")  # 分节线
    print(f"演示脚本运行（{len(demo_questions)} 题，Nudge 间隔={nudge_interval}）")  # 分节标题（80 题）
    print("─" * 60)  # 分节线

    # iters 按题数计数（与 demo_script 的 "nudge_interval" 对齐）
    # block_failed_turns 仅累积本块内的失败样本，只有这些会送入 Reviewer
    iters = 0  # 块内计数器
    block_correct = 0  # 块内对数
    block_total = 0  # 块内总数
    block_failed_turns: list[dict] = []  # 块内失败样本
    prev_probe_acc = None   # Reverse Eval Gate 锚点（上一次 Probe 准确率）

    for item in demo_questions:  # 遍历 80 题脚本
        seq = item["seq"]  # 脚本内序号
        eval_id = item["eval_id"]  # 关联评估集题号
        question = item["question"]  # 问题
        block = item.get("block", "")  # 所属块名

        answer = agent.answer(question)  # Agent 回答（动态挂载检索）
        ok, reason = evaluator.evaluate_answer(answer, eval_id)  # 三层判分
        block_correct += int(ok)  # 块内对数累计
        block_total += 1  # 块内总数累计
        iters += 1  # 计数器 +1
        if not ok:  # 答错/推脱
            block_failed_turns.append({"question": question, "answer": answer, "fail_reason": reason})  # 收集失败样本

        status = "✓" if ok else "✗"  # 对错符号
        note = f"  [{item['note']}]" if item.get("note") else ""  # 脚本备注（如"故意的边界题"）
        print(f"  Q{seq:02d} {status}  {question[:50]:<50}{note}")  # 打印逐题结果

        if iters >= nudge_interval:  # 到达 Nudge 间隔（每 10 题）
            block_acc = block_correct / block_total if block_total else 0  # 本块准确率
            print(f"\n{'━'*60}")  # 重分隔线
            print(f"  本块 [{block}] 完成: {block_correct}/{block_total} = {block_acc:.1%}")  # 打印块成绩

            # 全对 → 直接跳过 Nudge 和 Probe（零失败无需进化）
            if not block_failed_turns:  # 本块全对
                print("  ✓ 本块全对，跳过 Nudge 和 Probe eval")  # 提示跳过
                elog.add_nudge_event(seq, block, [], block_acc, sm.get_active_versions())  # 记空事件
            else:  # 有失败样本 → 触发 Nudge 复盘
                print(f"  🔔 Nudge 触发（{len(block_failed_turns)} 条失败样本注入 Reviewer）")  # 提示触发
                actions = reviewer.review(block_failed_turns)  # 复盘器生成动作（LLM）
                executed_actions = []  # 实际执行成功的动作
                for act in (actions or []):  # 逐个执行
                    try:  # 单动作失败不影响其余
                        # 安全红线守卫：与医疗安全政策冲突的进化动作直接阻断
                        violations = redline_check_action(act)  # 红线扫描
                        if violations:  # 命中红线
                            print(f"  [红线阻断] {act.get('skill_name','?')}: "
                                  f"{violations[0]['message']}")  # 打印阻断原因
                            continue  # 跳过该动作
                        if act["action"] == "create":  # 新建
                            ok_act = sm.create(act["skill_name"], act["content"], reason=act.get("reason",""))  # 落盘
                        elif act["action"] == "patch":  # 修补
                            ok_act = sm.patch(act["skill_name"], act["old_text"], act["new_text"], reason=act.get("reason",""))  # 落盘
                        else:  # 未知类型
                            ok_act = False  # 视为失败
                        if ok_act:  # 执行成功
                            executed_actions.append({"action": act["action"], "skill": act["skill_name"], "reason": act.get("reason","")[:80]})  # 记入已执行
                    except Exception as e:  # 执行异常
                        print(f"  [Reviewer] 执行失败: {e}")  # 打印错误
                print(f"  ✓ 执行了 {len(executed_actions)} 个 Skill 操作")  # 打印动作数
                if executed_actions:  # 有成功动作
                    store.reindex_all(sm.load_all())   # Skill 变更 → 重建向量索引

                probe_run_id = f"after_nudge_seq{seq}"  # Probe 运行标识
                probe_label = f"Nudge后（seq={seq}, block={block}）"  # Probe 标签
                probe_result = run_probe_eval(agent, evaluator, probe_ids, probe_run_id, probe_label, sm)  # 跑哨兵题回测
                probe_acc = probe_result['summary']['accuracy']  # 本次 Probe 准确率
                # Reverse Eval Gate：进化后准确率显著劣化 → 回滚本轮 Skill 变更
                if (executed_actions and prev_probe_acc is not None
                        and probe_acc < prev_probe_acc - 0.10):  # 劣化超 10 个百分点
                    print(f"  ⚠ Reverse Eval Gate 触发：{probe_acc:.1%} 低于锚点 "
                          f"{prev_probe_acc:.1%}，回滚本轮 {len(executed_actions)} 个操作")  # 打印回滚决策
                    for act in reversed(executed_actions):  # 逆序回滚
                        try:
                            sm.rollback(act["skill"], reason="reverse_eval_gate")  # 回滚单个技能
                        except Exception as e:  # 回滚失败
                            print(f"  [Rollback] {act['skill']} 失败: {e}")  # 打印错误
                    executed_actions = []  # 清空（本轮进化作废）
                    store.reindex_all(sm.load_all())  # 重建索引（恢复旧知识）
                    probe_result = run_probe_eval(agent, evaluator, probe_ids,
                                                  probe_run_id + "_rollback", probe_label, sm)  # 复测确认恢复
                    probe_acc = probe_result['summary']['accuracy']  # 复测准确率
                else:  # 进化保留
                    prev_probe_acc = probe_acc  # 更新锚点
                elog.add_eval_run(probe_result)  # 记入日志
                print(f"  Probe eval: {probe_result['summary']['correct']}/{probe_result['summary']['total']} = {probe_acc:.1%}")  # 打印 Probe 结果
                elog.add_nudge_event(seq, block, executed_actions, block_acc, sm.get_active_versions())  # 记 Nudge 事件

            # 重置块状态
            iters = 0  # 计数器归零
            block_correct = 0  # 对数归零
            block_total = 0  # 总数归零
            block_failed_turns = []  # 失败样本清空
            agent.conversation_history = agent.conversation_history[-5:]  # 只留最近 5 条对话（控制复盘输入规模）
            print(f"{'━'*60}\n")  # 分隔收尾

    # ── 最终评估 ─────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")  # 分节线
    print("最终评估（进化后 Skills）")  # 分节标题
    print("─" * 60)  # 分节线
    final = run_full_eval(agent, evaluator, "final", "最终（进化后）", sm)  # 再跑 60 题（约 93%）
    elog.add_eval_run(final)  # 记入日志
    _print_summary(final)  # 打印摘要

    # ── 汇总 ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")  # 汇总分隔线
    print("  实验结果汇总")  # 汇总标题
    print("=" * 60)  # 汇总分隔线
    base_acc = baseline["summary"]["accuracy"]  # 基线准确率
    final_acc = final["summary"]["accuracy"]  # 终局准确率
    print(f"  基线准确率:   {base_acc:.1%}  ({baseline['summary']['correct']}/{baseline['summary']['total']})")  # 打印基线
    print(f"  进化后准确率: {final_acc:.1%}  ({final['summary']['correct']}/{final['summary']['total']})")  # 打印终局
    print(f"  准确率提升:   +{(final_acc - base_acc):.1%}")  # 打印增益（33%→93% 的核心数字）

    print("\n  进化轨迹（probe eval 准确率）:")  # 轨迹标题
    for run in elog.eval_runs:  # 遍历各轮评估
        if run["run_id"].startswith("after_nudge"):  # 只看 Nudge 后的 Probe
            print(f"    {run['label']:<35}  {run['correct']:>2}/{run['total']:>2} = {run['accuracy']:.1%}")  # 打印每轮轨迹

    print("\n  Skill 版本历史:")  # 版本标题
    for name, versions in sorted(sm.get_all_version_summaries().items()):  # 按技能名遍历
        print(f"    {name}: {len(versions)} 个版本")  # 技能与版本数
        for v in versions:  # 遍历每条版本
            print(f"      v{v.get('time','')[:19]} [{v['action']}] {v['reason'][:60]}")  # 打印时间/动作/原因

    elog.save(sm, evaluator)  # 写 evolution_log.json
    print("\n✓ 评估详情见 outputs/eval_runs/")  # 提示归档位置
    print("✓ 各版本 Skill 见 outputs/skill_snapshots/")  # 提示快照位置
    print("\n  —— 双模态评估器统计 ——")  # 评估器统计标题
    print(f"  {evaluator.stats}")  # 打印三层评估统计（Judge 调用/翻案等）
    biz = ROOT / "data" / "business_metrics.json"  # 商业指标文件
    if biz.exists():  # 文件存在
        m = json.loads(biz.read_text(encoding="utf-8"))  # 读取声明口径指标
        print("\n  —— 商业化试点指标 ——")  # 商业指标标题
        for mt in m["metrics"]:  # 遍历指标条目
            print(f"  {mt['name']}: {mt['value']}（基线 {mt['baseline']}）")  # 打印指标与基线对照


def _print_summary(result: dict):
    r = result["summary"]  # 取汇总
    print(f"总体准确率: {r['correct']}/{r['total']} = {r['accuracy']:.1%}")  # 打印总体
    print("分类准确率:")  # 分节标题
    for cat, stats in sorted(result["by_category"].items()):  # 按类目排序
        bar = "█" * int(stats["accuracy"] * 20)  # 文本柱状图
        print(f"  {cat:<22} {stats['correct']:>2}/{stats['total']:>2}  {bar} {stats['accuracy']:.0%}")  # 对齐打印


if __name__ == "__main__":  # 直接运行本脚本时
    if not os.getenv("DEEPSEEK_API_KEY"):  # 未配置 Key
        print("错误: 请先设置 DEEPSEEK_API_KEY 环境变量")  # 报错提示
        sys.exit(1)  # 退出（实验必须真实调 LLM）
    run_experiment()  # 启动完整实验
