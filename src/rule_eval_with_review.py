"""
用当前 skills/ 目录下的 Skill 跑一次全量规则评估（60题），
保存答案到 outputs/rule_eval_full.json，便于后续复核。
"""
import sys
import json  # os 未直接用但保留导入；sys 用于注入模块路径；json 用于结果落盘
from pathlib import Path  # 计算项目根目录与输出文件路径
from datetime import datetime  # 结果文件打时间戳

ROOT = Path(__file__).parent.parent  # 项目根目录（本文件在 src/ 下）
sys.path.insert(0, str(ROOT / "src"))  # 确保 src/ 内模块可按顶级名导入（直接运行本脚本时需要）

from skill_manager import SkillManager  # Skill 文件管理器：读取当前技能集
from evaluator import Evaluator  # 评估器：加载 60 题评估集并判分
from agent import CustomerServiceAgent  # 客服 Agent：用当前技能回答问题

# ═════════════════════════════════════════════════════════════════
# 脚本级演示：SkillManager + Agent + Evaluator 三件套组装
# 对应简历口径"60 题评估集"的离线复测入口（不经流水线，纯规则判分）
# ═════════════════════════════════════════════════════════════════
sm = SkillManager(str(ROOT / "skills"), str(ROOT / "outputs" / "skill_versions"))  # 技能目录 + 版本历史目录
ev = Evaluator(str(ROOT / "data" / "eval_set.json"))  # 加载评估集（60 题：问题+标准答案+关键词）
agent = CustomerServiceAgent(sm, nudge_interval=0)  # nudge_interval=0：本脚本不做后台复盘，只跑评估

print(f"当前 Skills: {list(sm.load_all().keys())}")  # 打印当前挂载的技能清单（进化到什么程度）
print(f"共 {len(ev.questions)} 题\n")  # 打印评估集题量

results = []  # 逐题明细（问题/答案/判分），最终落盘供复核
correct = 0  # 累计答对题数
by_cat = {}  # 按类目聚合的统计 {类目: {total, correct}}

for qid in sorted(ev.questions.keys()):  # 按题号顺序遍历评估集
    q = ev.questions[qid]  # 取该题（question/ground_truth/category/关键词）
    ans = agent.answer(q["question"])  # Agent 用当前技能生成回答（会触发检索挂载）
    # 逐题评分：答案经 evaluate_answer 双模态判定（保留 60 题完整结果供复核）
    ok, reason = ev.evaluate_answer(ans, qid)  # 规则关键词硬闸 + 语义余弦 → (是否正确, 判分理由)
    if ok:  # 答对则计数
        correct += 1
    cat = q["category"]  # 该题所属类目
    by_cat.setdefault(cat, {"total": 0, "correct": 0})  # 类目首次出现则初始化统计桶
    by_cat[cat]["total"] += 1  # 类目总题数 +1
    if ok:  # 类目答对数 +1
        by_cat[cat]["correct"] += 1

    mark = "✓" if ok else "✗"  # 对/错标记符号
    print(f"Q{qid:02d} {mark} [{cat}] {q['question'][:50]}")  # 打印题号、判定、类目与问题摘要
    if not ok:  # 答错的题
        print(f"      → {reason}")  # 额外打印判错理由（失败样本即复盘输入）

    results.append({  # 记录该题完整明细
        "id": qid,  # 题号
        "category": cat,  # 类目
        "question": q["question"],  # 原问题
        "ground_truth": q["ground_truth"],  # 标准答案（复核对照）
        "answer": ans,  # Agent 实际回答
        "rule_correct": ok,  # 规则判定结果
        "rule_reason": reason,  # 规则判定理由
    })

print(f"\n总体准确率: {correct}/{len(ev.questions)} = {correct/len(ev.questions):.1%}")  # 打印总准确率（如 56/60 = 93.3%）
print("分类准确率:")  # 表头
for cat in sorted(by_cat):  # 按类目名排序输出
    s = by_cat[cat]  # 该类目统计
    s["accuracy"] = round(s["correct"]/s["total"], 3)  # 计算并存类目准确率（3 位小数）
    print(f"  {cat:<22} {s['correct']:>2}/{s['total']:>2}  {s['accuracy']:.0%}")  # 对齐打印：类目 对/总 百分比

out = ROOT / "outputs" / "rule_eval_full.json"  # 完整结果输出文件路径
out.write_text(json.dumps({  # 组装顶层 JSON 并写盘
    "timestamp": datetime.now().isoformat(),  # 本次评估时间戳
    "skill_versions_active": sm.get_active_versions(),  # 本次生效的技能版本快照（溯源）
    "summary": {"total": len(ev.questions), "correct": correct,  # 汇总：题数与答对数
                "accuracy": round(correct/len(ev.questions), 3)},  # 汇总准确率
    "by_category": by_cat,  # 分类统计
    "results": results,  # 逐题明细
}, ensure_ascii=False, indent=2), encoding="utf-8")  # 中文不转义、缩进 2 空格，便于人工复核
print(f"\n✓ 完整结果已保存到 {out}")  # 提示结果文件位置
