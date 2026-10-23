"""
教学演示服务：步进式 Agent 自进化 UI（异步任务解耦版 v3）

启动：
  cd bp_monitor_service_agent
  uvicorn serve:app --host 0.0.0.0 --port 8000 --reload

架构（对比旧版 asyncio.to_thread 长事务）：
  POST /job/{baseline|block|final} → 校验后立即返回 job_id（后台队列执行）
  GET  /stream/events?job_id=...  → SSE 增量推流进度事件（轮询事件日志）
  耗时评估 / Nudge 进化在后台 Worker（线程或 Celery 进程）中执行，
  多个浏览器并发打开互不阻塞；任务串行化保证实验状态一致。

可选扩展（Celery）：
  set TASK_BACKEND=celery && set CELERY_BROKER_URL=redis://localhost:6379/0
  celery -A src.celery_app worker --loglevel=info --pool=solo
"""

import sys
import asyncio
import shutil  # os/sys 预留；asyncio 事件循环与 to_thread；shutil 目录还原
from pathlib import Path  # 定位项目根目录与输出目录
from contextlib import asynccontextmanager  # FastAPI lifespan 生命周期管理

from fastapi import FastAPI  # Web 框架
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse  # SSE 流/静态页/JSON 响应
from fastapi import HTTPException  # 抛 409 等业务错误

ROOT = Path(__file__).parent  # 项目根目录（本文件在根目录）
sys.path.insert(0, str(ROOT / "src"))  # 让 src 内模块可按顶级名导入

from task_queue import TaskManager  # 任务队列门面（inprocess/celery 可插拔）
from pipeline import (SKILLS_DIR, SKILLS_ORIG, load_state, save_state,
                      get_components, _load_business_metrics)  # 复用流水线的路径/状态/组件/商业指标

tm = TaskManager()  # 全局任务管理器（按 TASK_BACKEND 选择后端）
_active_job: dict = {}   # task_key -> job_id（防重复提交；终态后自动清理）


# ── 初始化 / 还原 ─────────────────────────────────────────────────────────────

# ═════════════════════════════════════════════════════════════════
# _restore：重置实验现场——从原始备份还原技能目录，
# 并清理版本/快照/评估/任务日志等输出目录，保证实验可重复
# ═════════════════════════════════════════════════════════════════
def _restore():
    """恢复初始 Skill 并清理版本/快照/评估/任务日志等输出目录。

    保证每次实验都从相同起点开始、结果可重复；Windows 沙箱禁止删目录时，
    降级为逐文件覆盖 SKILL.md，避免退出码中断启动。"""
    if SKILLS_ORIG.exists():  # 有原始备份才需要还原
        try:  # 首选：整树删除再复制
            if SKILLS_DIR.exists():  # 活动技能目录存在
                shutil.rmtree(SKILLS_DIR)  # 删除（含进化产物）
            shutil.copytree(SKILLS_ORIG, SKILLS_DIR)  # 从备份复制回种子
        except PermissionError:  # Windows 沙箱可能禁止删除目录
            # Windows 沙箱可能禁止删除目录，改为逐文件覆盖
            for orig_file in SKILLS_ORIG.rglob("SKILL.md"):  # 遍历备份里的每个 SKILL.md
                rel = orig_file.relative_to(SKILLS_ORIG)  # 计算相对路径
                dest = SKILLS_DIR / rel  # 目标路径
                dest.parent.mkdir(parents=True, exist_ok=True)  # 确保父目录存在
                dest.write_text(orig_file.read_text(encoding="utf-8"), encoding="utf-8")  # 覆盖写回
    for d in [ROOT / "outputs" / "skill_versions", ROOT / "outputs" / "skill_snapshots",
              ROOT / "outputs" / "eval_runs", ROOT / "outputs" / "jobs"]:  # 需清空的四个输出目录
        try:  # 逐目录删除（单个失败不影响其他）
            if d.exists():  # 目录存在
                shutil.rmtree(d)  # 删除
        except PermissionError:  # 权限不足
            pass  # 静默跳过（不中断启动）


# ═════════════════════════════════════════════════════════════════
# lifespan：应用生命周期——启动时备份种子技能、预热组件、写初始状态
# ═════════════════════════════════════════════════════════════════
@asynccontextmanager  # FastAPI lifespan 协议（yield 前后分别是启停逻辑）
async def lifespan(app: FastAPI):
    """应用生命周期：首次启动备份初始 Skills、预热组件并写入初始实验状态。"""
    (ROOT / "outputs").mkdir(exist_ok=True)  # 确保 outputs 目录存在
    if not SKILLS_ORIG.exists():  # 首次启动（无原始备份）
        shutil.copytree(SKILLS_DIR, SKILLS_ORIG)  # 备份种子技能（永不覆盖）
    # 预热组件（Skill 索引、嵌入器、评估器），并写入初始实验状态
    components = get_components()  # 懒加载全部组件（含建索引）
    state = load_state()  # 读状态（可能已有历史）
    state.setdefault("phase", "idle")  # 无 phase 字段时补 idle
    save_state(state)  # 落盘
    print(f"  [serve] 组件就绪: eval_mode={components['ev'].mode}, "
          f"embedder={components['store'].embed.name}, "
          f"vector_backend={components['store'].store.backend}, "
          f"task_backend={tm.impl.backend}")  # 打印运行形态摘要（评估/嵌入/向量/任务后端）
    yield  # 应用运行期（请求都在此后处理）


app = FastAPI(lifespan=lifespan)  # 创建 FastAPI 应用（挂生命周期）

SSE_HEADERS = {  # SSE 响应头
    "Cache-Control": "no-cache",  # 禁缓存（事件必须实时）
    "X-Accel-Buffering": "no",  # 禁 Nginx 代理缓冲
    "Access-Control-Allow-Origin": "*",  # 允许跨域（本地演示前端）
}


def _evt(data: dict) -> str:
    """把一个事件 dict 序列化为 SSE 帧（`data: {...}` 后跟空行）。"""
    return f"data: {json_dumps(data)}\n\n"  # SSE 协议：data 行 + 空行结束一帧


def json_dumps(obj) -> str:
    """JSON 序列化（保留中文原样，不转义为 \\uXXXX）。"""
    import json  # 局部导入（保持模块顶部轻量）
    return json.dumps(obj, ensure_ascii=False)  # 中文原样输出


# ── Job 防重 / 状态辅助 ───────────────────────────────────────────────────────

# ═════════════════════════════════════════════════════════════════
# _cleanup_finished_jobs：清掉已终态（done/failed/unknown）的任务
# 占位，让 _active_job 只保留"仍在运行"的映射
# ═════════════════════════════════════════════════════════════════
def _cleanup_finished_jobs():
    for key, job_id in list(_active_job.items()):  # 拷贝键列表再遍历（边遍历边删安全）
        if tm.get(job_id)["status"] in ("done", "failed", "unknown"):  # 任务已终态
            del _active_job[key]  # 释放该任务键（允许再次提交同类任务）


# ═════════════════════════════════════════════════════════════════
# _submit：任务提交统一入口——阶段门禁校验 → 防重复 → 入队 → 返回 job_id
# HTTP 请求毫秒级返回，真正耗时的执行在后台 Worker
# ═════════════════════════════════════════════════════════════════
def _submit(task: str, key: str, params: dict | None = None) -> dict:
    """校验阶段 → 入队后台任务 → 立即返回 job_id（HTTP 请求不等待执行）。"""
    _cleanup_finished_jobs()  # 先清理已结束任务
    state = load_state()  # 读实验状态（门禁依据）
    phase = state["phase"]  # 当前阶段

    # 阶段门禁（与旧版一致：按实验流程顺序触发）
    if key == "baseline" and phase != "idle":  # 基线只能从 idle 开始
        raise HTTPException(409, f"基线评估仅能在 idle 阶段提交，当前: {phase}")  # 拒绝
    if key == "block":  # 块任务：序号必须与状态指针一致
        idx = params["block_index"]  # 请求的块序号
        blocks = get_components()["blocks"]  # 全部块（8 块）
        if idx != state["current_block"] or idx >= len(blocks):  # 跳块/重复/越界
            raise HTTPException(409, f"需要运行第 {state['current_block']} 块，收到 {idx}")  # 拒绝
    if key == "final" and phase != "all_blocks_done":  # 终局必须在全部块完成后
        raise HTTPException(409, f"最终评估需在 all_blocks_done 阶段提交，当前: {phase}")  # 拒绝

    # 防重复提交：同类任务还在跑就不允许再提（多用户并发点同一个按钮）
    if key in _active_job:  # 同类任务运行中
        raise HTTPException(409, f"任务 {key} 正在运行（job={_active_job[key]}），请等待完成")  # 拒绝

    job_id = tm.submit(task, params or {})  # 入队（立即返回 job_id）
    _active_job[key] = job_id  # 登记占用该任务键
    return {"job_id": job_id, "task": task, "phase": phase}  # 返回提交回执（前端拿 job_id 订阅 SSE）


# ── Job 提交接口（提交即返回，非阻塞） ────────────────────────────────────────

@app.post("/job/baseline")
async def job_baseline():
    """提交基线评估任务（入队即返回 job_id）。"""
    return _submit("baseline", "baseline")  # 提交基线（60 题初始评估）


@app.post("/job/block/{block_index}")
async def job_block(block_index: int):
    """提交指定块的进化评估任务（入队即返回 job_id）。"""
    return _submit("block", "block", {"block_index": block_index})  # 提交第 N 块（路径参数传入）


@app.post("/job/final")
async def job_final():
    """提交最终评估任务（入队即返回 job_id）。"""
    return _submit("final", "final")  # 提交终局评估


# 固定路径路由必须注册在动态路由 /job/{job_id} 之前，
# 否则 "active" 会被当作 job_id 吞掉（路由遮蔽，接口变成死代码）
@app.get("/job/active")
async def job_active():
    """返回当前仍在运行的任务映射（task_key -> job_id）。"""
    _cleanup_finished_jobs()  # 先清终态
    return JSONResponse(_active_job)  # 返回运行中任务（前端按钮禁用依据）


@app.get("/job/{job_id}")
async def job_status(job_id: str):
    """查询单个 job 的状态与结果。"""
    return JSONResponse(tm.get(job_id))  # 透传任务状态（pending/running/done/failed）


# ── SSE 事件流（增量拉取 job 事件日志，端到端无阻塞） ─────────────────────────

# ═════════════════════════════════════════════════════════════════
# /stream/events：SSE 增量推流——以 after_seq 为游标轮询事件日志，
# 每次只取新增事件推给前端；轮询在线程池执行（to_thread）不阻塞事件循环
# ═════════════════════════════════════════════════════════════════
@app.get("/stream/events")
async def stream_events(job_id: str, after_seq: int = 0):
    """SSE 增量推流：以 after_seq 为游标拉取事件，游标单调推进直到任务终态。"""
    async def gen():  # 异步生成器：逐帧产出 SSE 事件
        seq = after_seq  # 游标从请求参数起步（断线重连可续传）
        while True:  # 轮询循环
            events, job = await asyncio.to_thread(tm.poll_events, job_id, seq)  # 线程池里拉增量事件（阻塞调用不卡事件循环）
            for e in events:  # 遍历新事件
                seq = max(seq, e["seq"])  # 游标单调推进到最大序号
                yield _evt(e)  # 序列化为 SSE 帧推给前端
            status = job["status"]  # 当前任务状态
            if status in ("done", "failed", "unknown"):  # 任务终态
                if status == "failed":  # 失败：额外推错误帧
                    yield _evt({"seq": -1, "type": "error",
                                "message": f"任务执行失败: {job.get('error', '')[:200]}"})  # 错误信息截断 200 字
                yield _evt({"seq": -1, "type": "done"})  # 推送结束帧（前端关闭 EventSource）
                break  # 退出轮询
            await asyncio.sleep(0.3)  # 未终态：歇 300ms 再拉（平衡实时性与空转）

    return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)  # 以 SSE 流返回


# ── 静态 / 状态接口 ───────────────────────────────────────────────────────────

@app.get("/")
async def index():
    """返回前端入口页面。"""
    return FileResponse(ROOT / "index.html")  # 返回三栏演示界面 HTML


# ═════════════════════════════════════════════════════════════════
# /state：状态聚合接口（前端刷新/断线恢复的锚点）
# 汇合实验进度、各轮评估结果、系统能力、检索观测、实测指标、审计摘要
# ═════════════════════════════════════════════════════════════════
@app.get("/state")
async def get_state():
    """聚合实验状态 + 系统能力 + 运行时指标 + 审计摘要，供前端面板渲染。"""
    _cleanup_finished_jobs()  # 清理终态任务（active_jobs 数据干净）
    state = load_state()  # 读实验状态
    c = get_components()  # 取组件（拿系统能力信息）
    blocks = c["blocks"]  # 全部块
    return JSONResponse({  # 聚合响应
        "phase": state["phase"],  # 当前阶段（idle/baseline_running/.../complete）
        "current_block": state["current_block"],  # 当前块指针
        "total_blocks": len(blocks),  # 总块数（8）
        "nudge_count": state.get("nudge_count", 0),  # Nudge 累计次数
        "block_names": [b[0]["block"] for b in blocks],  # 各块名称（每块取首题的 block 字段）
        "eval_results": {  # 各轮评估结果（压缩为关键指标）
            k: {"accuracy": v["accuracy"], "correct": v["correct"], "total": v["total"],
                "by_category": v["by_category"]}
            for k, v in state["eval_results"].items()  # baseline/after_block_N/final
        },
        # v3 新增：系统能力面板数据
        "system_info": {  # 系统能力信息
            "eval_mode": c["ev"].mode,  # 评估模式
            "judge_model": c["ev"].judge.model,  # Judge 模型名
            "judge_available": c["ev"].judge.available,  # Judge 是否可用
            "judge_vote": c["ev"].vote_n,  # 投票次数
            "embedder": c["store"].embed.name,  # 嵌入器名
            **c["store"].stats(),  # 向量库统计（后端/分块数/维度）
            "task_backend": tm.impl.backend,  # 任务后端
            "active_jobs": _active_job,  # 运行中任务
        },
        "retrieval_obs": state.get("retrieval_obs", {}),  # 检索命中率观测
        "business_metrics": _load_business_metrics(),  # 声明口径商业指标
        # v3.1 新增：实测化的运行时商业指标 + 审计/复核队列
        "runtime_metrics": c["metrics"].snapshot(),  # 实测指标（分流率/延迟/上线数）
        "audit_stats": {**c["audit"].stats(),  # 审计统计（累计条数）
                        "pending_risks": len(c["audit"].pending_risks(100))},  # 待复核高风险条数
        "eval_mode_hint": _eval_mode_hint(c),  # 评估模式中文说明
    })


def _eval_mode_hint(c: dict) -> str:
    """把评估模式枚举翻译成给前端展示的中文说明。"""
    m = c["ev"].mode  # 当前评估模式
    hints = {  # 模式 → 中文说明映射
        "rule": "纯关键词规则（可复现的确定性基线）",  # 纯规则
        "semantic": "规则 + 语义相似度记录（无 LLM 成本，不翻案）",  # 规则+语义
        "hybrid": "规则硬闸 + 语义相似度 + LLM Judge 翻案（当前）",  # 三层混合
        "llm": "仅 Judge 裁决（实验对照）",  # 纯 Judge
    }
    return hints.get(m, m)  # 找不到说明就原样返回模式名


@app.get("/skills")
async def get_skills():
    """返回所有 Skill 的当前内容与版本历史摘要。"""
    sm = get_components()["sm"]  # 取技能管理器
    result = {}  # 结果：技能名 → {content, version_count, history}
    for name, content in sorted(sm.load_all().items()):  # 按名遍历当前技能
        history = sm.get_version_history(name)  # 读版本历史
        result[name] = {  # 组装该技能信息
            "content": content,  # 当前全文
            "version_count": len(history),  # 版本数
            "history": [  # 历史摘要列表（去大字段）
                {"version": h.get("version", i + 1), "time": h["time"][:19],  # 版本号与时间（截到秒）
                 "action": h["action"], "reason": h["reason"][:120]}  # 动作与原因摘要
                for i, h in enumerate(history)  # 遍历历史（i 兜底版本号）
            ],
        }
    return JSONResponse(result)  # 返回全部技能信息


@app.get("/skill_version/{name}/{version}")
async def get_skill_version(name: str, version: int):
    """返回指定 Skill 的指定历史版本内容（供 UI 对比/快照查看）。"""
    history = get_components()["sm"].get_version_history(name)  # 读该技能历史
    for h in history:  # 查找目标版本
        if h.get("version") == version:  # 命中
            return JSONResponse({  # 返回该版本详情
                "content": h["content"], "action": h["action"],  # 全文与动作
                "reason": h["reason"], "time": h["time"][:19],  # 原因与时间
            })
    return JSONResponse({"error": "not found"}, status_code=404)  # 版本不存在


# ═════════════════════════════════════════════════════════════════
# /reset：重置实验——还原技能、清状态、重建索引；
# 有任务运行时拒绝（防止半途回滚造成状态错乱）
# ═════════════════════════════════════════════════════════════════
@app.post("/reset")
async def reset():
    """重置实验：还原初始 Skill、清空状态并重建向量索引（有任务在跑时拒绝）。"""
    _cleanup_finished_jobs()  # 清理终态任务
    if _active_job:  # 仍有任务在跑
        raise HTTPException(409, f"有任务正在运行 {_active_job}，请等待完成后再重置")  # 拒绝重置
    try:
        _restore()  # 还原技能目录 + 清理输出目录
    except Exception as e:  # 还原过程异常
        print(f"[reset] _restore warning: {e}")  # 只告警不中断（降级路径已在 _restore 内处理）
    state = {"phase": "idle", "current_block": 0, "nudge_count": 0,
             "eval_results": {}, "conv_history": []}  # 全新初始状态
    save_state(state)  # 落盘
    # 重建 Skill 向量索引（skills 已还原）
    c = get_components()  # 取组件
    c["store"].reindex_all(c["sm"].load_all())  # 用还原后的技能重建索引
    return {"status": "ok"}  # 返回成功


# ── 兼容旧端点：提交 + 原地订阅（老前端无需改动也能用） ──────────────────────

@app.get("/stream/baseline")
async def old_stream_baseline():
    """兼容旧版：提交基线任务并原地返回 SSE 流。"""
    info = _submit("baseline", "baseline")  # 提交任务
    return _legacy_stream(info["job_id"])  # 直接返回事件流


@app.get("/stream/block/{block_index}")
async def old_stream_block(block_index: int):
    """兼容旧版：提交块进化任务并原地返回 SSE 流。"""
    info = _submit("block", "block", {"block_index": block_index})  # 提交块任务
    return _legacy_stream(info["job_id"])  # 直接返回事件流


@app.get("/stream/final")
async def old_stream_final():
    """兼容旧版：提交最终评估任务并原地返回 SSE 流。"""
    info = _submit("final", "final")  # 提交终局任务
    return _legacy_stream(info["job_id"])  # 直接返回事件流


# ═════════════════════════════════════════════════════════════════
# _legacy_stream：与 /stream/events 相同的推流逻辑（游标从 0 起），
# 供旧端点"提交+订阅"一步式使用
# ═════════════════════════════════════════════════════════════════
def _legacy_stream(job_id: str):
    """与 /stream/events 相同的事件流，直接内联。"""
    async def gen():  # 异步生成器（与新版唯一的差别：游标从 0 开始）
        seq = 0  # 从头推全部事件
        while True:  # 轮询循环
            events, job = await asyncio.to_thread(tm.poll_events, job_id, seq)  # 线程池拉增量
            for e in events:  # 遍历新事件
                seq = max(seq, e["seq"])  # 推进游标
                yield _evt(e)  # 推帧
            status = job["status"]  # 任务状态
            if status in ("done", "failed", "unknown"):  # 终态
                if status == "failed":  # 失败
                    yield _evt({"type": "error",
                                "message": f"任务执行失败: {job.get('error', '')[:200]}"})  # 错误帧
                yield _evt({"type": "done"})  # 结束帧
                break  # 退出
            await asyncio.sleep(0.3)  # 轮询间隔

    return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)  # SSE 流返回
