"""
离线评估门禁（P2）：无 API Key、无外部服务、确定性可复现的发布前自检。

定位与设计：
  CI 的 lint/test 只能证明"代码没坏"，无法证明"评估器还判得对"。
  本门禁在 docker 构建前加一道语义闸门（ci.yml 的 eval-gate job）——
  任何 PR 改动评估器规则/Skill 结构导致判别力回退时，门禁失败阻断合入。

三类检查（全程离线、确定性、不改动 outputs/ 与实验 state 现场）：
  1. 结构校验：
     - skills/ 下每个技能目录的 SKILL.md 存在且非空（Agent 知识库完整性）
     - EventLog（src/task_queue.py）在 tempfile.TemporaryDirectory 里写入
       3 条事件后读回，校验 seq 严格递增、每条含 ts/type 字段、
       JSONL 每行可 json.loads（SSE 增量推流的游标语义依赖）
  2. 判别力校验：golden_pairs.json（6 对好答案 expect_pass=true +
     6 对坏答案 expect_pass=false，覆盖 6 个类目）逐对调用评估器离线
     打分，实测 good_pass_rate（好对被判通过的比例）与 bad_block_rate
     （坏对被判不通过的比例），任一低于 thresholds.json 阈值即 FAIL
  3. /metrics 冒烟：TestClient（不进 lifespan）GET /metrics 200 且
     文本含 http_requests_total（可观测基建最小闭环）

评估器直接调用路径的副作用核实：Evaluator.evaluate_answer() 只更新
内存中的 self.stats / self.last_detail / self._judge_cache，不落盘、
不触碰 outputs/ 与 experiment_state.json —— 门禁可以安全直调，
无需换更底层入口。

环境固定：开头逐行复刻 tests/conftest.py 的离线环境配置（hash 嵌入 +
inprocess 队列 + rule 评估 + 关闭 Milvus/Judge Key），保证独立运行
（python evals/gate.py）与 pytest 环境完全一致、结果可复现。

用法：
  python evals/gate.py     # 全部 PASS 退出码 0；任一 FAIL 退出码 1（CI 据此阻断）
"""

import os  # 环境变量固定（必须在导入任何 src/serve 模块之前执行）

# ── 离线环境固定（复刻 tests/conftest.py；强制赋值：门禁确定性优先于宿主机环境）──
os.environ["EMBED_PROVIDER"] = "hash"          # hash 嵌入：确定性、零下载、零 API
os.environ["TASK_BACKEND"] = "inprocess"       # 进程内队列：不需要 Redis
os.environ["EVAL_MODE"] = "rule"               # 纯规则评估：不需要 LLM Key（无 Key 时 auto 会降级 semantic，门禁直接钉死 rule 保证确定性）
os.environ["MILVUS_ENABLED"] = "0"             # 关闭 Milvus（本地向量库即可）
os.environ["MILVUS_URI"] = ""                  # 清空连接地址
os.environ["BP_EMBED_MODEL_DIR"] = ""          # 禁用本地模型目录
os.environ["EMBED_API_KEY"] = ""               # 清空嵌入 API Key
os.environ["DEEPSEEK_API_KEY"] = ""            # 清空主 LLM Key
os.environ["JUDGE_API_KEY"] = ""               # 清空 Judge Key
os.environ.setdefault("RATE_LIMIT_RPM", "0")   # 限流默认关闭（对齐 conftest；/metrics 本身也限流豁免）

import sys  # noqa: E402  导入路径注入须在环境固定之后
import json  # noqa: E402  金标对/阈值/JSONL 解析
import tempfile  # noqa: E402  EventLog 结构校验的临时目录（用完即毁）
from pathlib import Path  # noqa: E402  定位项目根与 evals 配置文件

ROOT = Path(__file__).parent.parent  # 项目根目录（本文件在 evals/ 下）
sys.path.insert(0, str(ROOT / "src"))  # src 内模块按顶级名导入（对齐 serve.py 约定）
sys.path.insert(0, str(ROOT))  # serve 按顶级名导入

import task_queue as tq  # noqa: E402  EventLog 所在模块（JOBS_DIR 重定向用）
import pipeline  # noqa: E402  SKILLS_DIR / EVAL_SET 路径常量
import serve  # noqa: E402  /metrics 冒烟对象（导入即注册中间件；不进 lifespan）
from evaluator import Evaluator  # noqa: E402  判别力校验的评估器
from fastapi.testclient import TestClient  # noqa: E402  内存内 HTTP 客户端

GATE_DIR = Path(__file__).parent  # evals/ 目录（金标对与阈值文件所在）
GOLDEN_PAIRS_FILE = GATE_DIR / "golden_pairs.json"  # 判别力金标对（12 对）
THRESHOLDS_FILE = GATE_DIR / "thresholds.json"  # 判别力阈值（实测值 + 安全余量）


# ═════════════════════════════════════════════════════════════════
# 检查①：技能结构 —— SKILLS_DIR 下每个技能目录的 SKILL.md 存在且非空
# ═════════════════════════════════════════════════════════════════
def check_skills_structure() -> tuple[bool, str]:
    """结构校验（技能库完整性）：SKILL.md 缺失/为空均判 FAIL。"""
    if not pipeline.SKILLS_DIR.exists():  # 技能根目录不存在
        return False, "skills/ 目录不存在"
    skill_dirs = [d for d in pipeline.SKILLS_DIR.iterdir() if d.is_dir()]  # 全部技能目录
    if not skill_dirs:  # 一个技能都没有
        return False, "skills/ 下没有任何技能目录"
    problems = []  # 收集各目录的问题
    for d in skill_dirs:  # 逐目录检查
        skill_file = d / "SKILL.md"  # 技能正文固定命名
        if not skill_file.exists():  # 文件缺失
            problems.append(f"{d.name}: 缺少 SKILL.md")
        elif not skill_file.read_text(encoding="utf-8").strip():  # 文件为空
            problems.append(f"{d.name}: SKILL.md 为空")
    if problems:  # 有问题 → FAIL（逐条列出）
        return False, "; ".join(problems)
    names = ", ".join(sorted(d.name for d in skill_dirs))  # 通过的目录清单
    return True, f"{len(skill_dirs)} 个技能目录 SKILL.md 均存在且非空（{names}）"


# ═════════════════════════════════════════════════════════════════
# 检查②：EventLog 结构 —— 临时目录写 3 条事件读回校验游标/字段/JSONL
# ═════════════════════════════════════════════════════════════════
def check_eventlog_integrity() -> tuple[bool, str]:
    """结构校验（事件日志完整性）：seq 严格递增 + ts/type 齐全 + JSONL 可解析。"""
    with tempfile.TemporaryDirectory() as tmp:  # 临时目录：不污染真实 outputs/jobs/
        saved_jobs_dir = tq.JOBS_DIR  # 保存原目录（校验完恢复）
        tq.JOBS_DIR = Path(tmp)  # 重定向事件日志落盘目录
        try:
            log = tq.EventLog("gate-smoke")  # 在临时目录里建事件日志
            log.append("job_queued", task="baseline")  # 事件 1：入队
            log.append("job_started", task="baseline")  # 事件 2：开始
            log.append("eval_complete", accuracy=0.5)  # 事件 3：完成
            events = log.since(0)  # 读回全部事件
            jsonl_path = Path(tmp) / "gate-smoke.jsonl"  # 落盘文件路径
            lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()  # 逐行读回 JSONL
        finally:
            tq.JOBS_DIR = saved_jobs_dir  # 恢复原目录（不影响后续任何逻辑）
        seqs = [e["seq"] for e in events]  # 实际序号序列
        if len(events) != 3 or seqs != [1, 2, 3]:  # 数量或严格递增被破坏
            return False, f"事件数/序号异常: {len(events)} 条, seq={seqs}"
        if not all("ts" in e and "type" in e for e in events):  # 必备字段缺失
            return False, "存在缺少 ts/type 字段的事件"
        if len(lines) != 3:  # 落盘行数与内存不符
            return False, f"JSONL 行数异常: {len(lines)}"
        try:  # 逐行解析 JSONL（崩溃恢复语义依赖每行独立可解析）
            for line in lines:
                json.loads(line)
        except json.JSONDecodeError as e:  # 有坏行
            return False, f"JSONL 存在不可解析行: {e}"
    return True, "3 条事件 seq 1→3 严格递增，均含 ts/type；JSONL 3 行全部可解析"


# ═════════════════════════════════════════════════════════════════
# 检查③：评估器判别力 —— 金标对逐对离线打分（rule 模式，零 LLM 成本）
# ═════════════════════════════════════════════════════════════════
def check_discrimination() -> dict:
    """判别力校验：返回 good_pass_rate / bad_block_rate 实测值与漏判明细。

    评估器直接调用 Evaluator.evaluate_answer()（纯内存判定，无落盘副作用，
    已核实），EVAL_MODE=rule 保证确定性：同输入永远同输出。"""
    pairs = json.loads(GOLDEN_PAIRS_FILE.read_text(encoding="utf-8"))["pairs"]  # 读金标对
    ev = Evaluator(str(pipeline.EVAL_SET), mode=os.getenv("EVAL_MODE", "rule"))  # rule 模式评估器
    good_total = good_hit = 0  # 好对计数：总数 / 被判通过数
    bad_total = bad_hit = 0  # 坏对计数：总数 / 被判拦截数
    misses = []  # 与预期不一致的明细（FAIL 时逐条展示）
    for p in pairs:  # 逐对打分
        ok, reason = ev.evaluate_answer(p["answer"], p["qid"])  # 离线判定（无副作用）
        if p["expect_pass"]:  # 好答案：期望通过
            good_total += 1
            if ok:  # 判通过 → 命中
                good_hit += 1
            else:  # 误杀
                misses.append(f"好对误杀 Q{p['qid']}: {reason}")
        else:  # 坏答案：期望拦截
            bad_total += 1
            if not ok:  # 判不通过 → 命中
                bad_hit += 1
            else:  # 漏判
                misses.append(f"坏对漏判 Q{p['qid']}: 回答='{p['answer'][:30]}...' 被判通过")
    return {  # 实测结果（通过与否由 main 里与阈值比较决定）
        "good_pass_rate": good_hit / good_total if good_total else 0.0,  # 好对通过率
        "bad_block_rate": bad_hit / bad_total if bad_total else 0.0,  # 坏对拦截率
        "good_detail": f"{good_hit}/{good_total}",  # 好对命中明细
        "bad_detail": f"{bad_hit}/{bad_total}",  # 坏对命中明细
        "misses": misses,  # 不一致明细
    }


# ═════════════════════════════════════════════════════════════════
# 检查④：/metrics 冒烟 —— 可观测基建最小闭环（TestClient 不进 lifespan）
# ═════════════════════════════════════════════════════════════════
def check_metrics_smoke() -> tuple[bool, str]:
    """/metrics 冒烟：200 + 文本含 http_requests_total。"""
    c = TestClient(serve.app)  # 不用 with：跳过 lifespan（避免 _restore 清空实验现场与组件预热）
    r = c.get("/metrics")  # 抓取指标端点（限流豁免路径）
    if r.status_code != 200:  # 端点不可用
        return False, f"/metrics 状态码 {r.status_code}"
    if "http_requests_total" not in r.text:  # 核心指标缺失
        return False, "/metrics 文本缺少 http_requests_total"
    return True, "/metrics 200，文本含 http_requests_total（content-type 含 text/plain）"


# ═════════════════════════════════════════════════════════════════
# main：逐项执行 → 打 PASS/FAIL → 汇总表 → 任一 FAIL 退出码 1
# ═════════════════════════════════════════════════════════════════
def main() -> int:
    thresholds = json.loads(THRESHOLDS_FILE.read_text(encoding="utf-8"))  # 读阈值配置
    rows = []  # 汇总表行：(检查项, 实测, 阈值, 结果)

    # ① 技能结构
    ok, detail = check_skills_structure()
    print(f"[{'PASS' if ok else 'FAIL'}] skills_structure: {detail}")
    rows.append(("skills_structure", "ok" if ok else "FAIL", "-", ok))

    # ② EventLog 结构
    ok, detail = check_eventlog_integrity()
    print(f"[{'PASS' if ok else 'FAIL'}] eventlog_integrity: {detail}")
    rows.append(("eventlog_integrity", "ok" if ok else "FAIL", "-", ok))

    # ③ 判别力（与阈值比较）
    stats = check_discrimination()
    t_good = thresholds["good_pass_rate"]["threshold"]  # 好对通过率阈值
    t_bad = thresholds["bad_block_rate"]["threshold"]  # 坏对拦截率阈值
    ok_good = stats["good_pass_rate"] >= t_good  # 实测 ≥ 阈值才算过
    ok_bad = stats["bad_block_rate"] >= t_bad
    print(f"[{'PASS' if ok_good else 'FAIL'}] good_pass_rate: "
          f"{stats['good_detail']} = {stats['good_pass_rate']:.2f} ≥ 阈值 {t_good}")
    print(f"[{'PASS' if ok_bad else 'FAIL'}] bad_block_rate: "
          f"{stats['bad_detail']} = {stats['bad_block_rate']:.2f} ≥ 阈值 {t_bad}")
    rows.append(("good_pass_rate", f"{stats['good_pass_rate']:.2f}", t_good, ok_good))
    rows.append(("bad_block_rate", f"{stats['bad_block_rate']:.2f}", t_bad, ok_bad))
    if stats["misses"]:  # 有不一致明细时逐条展示（便于定位回退原因）
        for m in stats["misses"]:
            print(f"       · {m}")

    # ④ /metrics 冒烟
    ok, detail = check_metrics_smoke()
    print(f"[{'PASS' if ok else 'FAIL'}] metrics_smoke: {detail}")
    rows.append(("metrics_smoke", "ok" if ok else "FAIL", "-", ok))

    # 汇总表
    print("\n──── 离线评估门禁汇总 ────")
    print(f"{'检查项':<22}{'实测':>8}{'阈值':>8}   结果")
    for name, measured, threshold, ok in rows:
        print(f"{name:<22}{str(measured):>8}{str(threshold):>8}   {'PASS' if ok else 'FAIL'}")

    all_ok = all(row[3] for row in rows)  # 任一项 FAIL 即整体 FAIL
    if all_ok:
        print("\n门禁通过（exit 0）")
        return 0
    print("\n门禁失败（exit 1）：请检查上方 FAIL 项——评估器判别力或结构校验出现回退")
    return 1


if __name__ == "__main__":
    sys.exit(main())  # CI 的 eval-gate job 据退出码判定成败
