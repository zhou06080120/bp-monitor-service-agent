"""
安全红线守卫：自进化闭环的护栏（Guardrail）。

背景：Reviewer 由 LLM 驱动，理论上可能生成与医疗安全政策冲突的 Skill 规则
（例如把"起搏器用户禁止测量"改写成"可以测量"）。本项目进化动作发生在
Skill 文件落盘前，必须经过规则化红线校验 —— 不依赖 LLM、确定性、可解释。

红线来源：data/redlines.json（正则列表）。本模块在 Skill create/patch 前
（或后、变更内容上）做匹配，命中即阻断并返回违规描述。

匹配语义：
  - 危险豁免型：「起搏器 用户 可以/没问题 测量」→ 禁止
  - 推责型：「无需就医 / 不用去医院 / 不必看医生」→ 禁止
  - 免责型：「不保证 / 不负责 / 无医疗风险」→ 禁止（客服必须转人工而非免责）

匹配仅在**新增文本**上执行（不含否定语境），命中即阻断；已存在的初始 Skill
内容若命中（理论不该发生）只告警、不静默修改。
"""

import json  # 解析 data/redlines.json 红线规则文件
import re  # 正则匹配：对 Skill 文本做危险模式扫描（确定性、可解释）
from pathlib import Path  # 定位项目根目录下的红线规则文件

ROOT = Path(__file__).parent.parent  # 项目根目录（本文件在 src/ 下，取上一级）
REDLINES_FILE = ROOT / "data" / "redlines.json"  # 红线规则文件路径（pattern+message 列表）

# 默认红线：即便 data/redlines.json 缺失也要生效的兜底
_BUILTIN_REDLINES = [  # 内置规则列表：文件缺失/损坏时的最终防线
    {"pattern": r"起搏器[^。\n]{0,15}(可以|能用|没问题|支持|适合)",  # 规则1：起搏器 + 15 字内出现许可词 → 危险豁免
     "message": "暗示起搏器用户可以使用电子血压计（政策红线：禁止使用）"},  # 命中时返回的违规说明
    {"pattern": r"(孕妇|房颤|心律不齐)[^。\n]{0,15}(可以|能用|没问题|放心|适合)",  # 规则2：高风险人群 + 许可词
     "message": "暗示高风险人群可无顾虑使用设备测量"},  # 违规说明（医疗安全）
    {"pattern": r"(无需|不用|不必|不用再)(就医|看医生|去医院|打120)",  # 规则3：劝阻就医类表述
     "message": "暗示用户无需就医（医疗风险提示红线）"},  # 违规说明
    {"pattern": r"(不负责|不保证|免责|无医疗风险|与公司无关)",  # 规则4：免责声明类表述
     "message": "包含免责声明（医疗器械客服不得推卸责任，应转人工）"},  # 违规说明（合规要求转人工）
    {"pattern": r"(拆除|拆卸|分解)[^。\n]{0,10}(电池|气泵|主板|传感器)",  # 规则5：引导用户自行拆机
     "message": "暗示用户自行拆解医疗器械内部部件"},  # 违规说明（器械安全）
]


# ═════════════════════════════════════════════════════════════════
# load_redlines：加载红线规则——优先外部配置文件，异常时内置规则兜底
# ═════════════════════════════════════════════════════════════════
def load_redlines() -> list[dict]:
    """加载红线规则：优先读 data/redlines.json，缺失/损坏时用内置兜底规则。"""
    if REDLINES_FILE.exists():  # 红线文件存在才尝试加载
        try:
            data = json.loads(REDLINES_FILE.read_text(encoding="utf-8"))  # 读取文件内容并解析为 JSON
            if isinstance(data, list):  # 合法格式应为规则列表
                return data  # 返回自定义红线规则（允许运营侧自行扩充）
        except json.JSONDecodeError:  # 文件损坏（非法 JSON）
            pass  # 静默降级，不中断主流程
    return _BUILTIN_REDLINES  # 文件缺失/损坏/格式不对 → 内置规则兜底


# ═════════════════════════════════════════════════════════════════
# check_text：对一段 Skill 文本逐条扫描红线正则，返回违规命中列表
# 输出 [{pattern, message, snippet}]；空列表 = 通过（可安全落盘）
# ═════════════════════════════════════════════════════════════════
def check_text(text: str, redlines: list[dict] | None = None) -> list[dict]:
    """对一段 Skill 内容做红线扫描，返回违规列表 [{pattern, message, snippet}]。"""
    hits = []  # 收集所有命中的违规项（一条文本可能命中多条规则）
    for r in (redlines or load_redlines()):  # 遍历规则（未显式传入则加载默认规则）
        m = re.search(r["pattern"], text)  # 在文本中搜索该危险模式（首个命中即可）
        if m:  # 命中红线
            s = max(0, m.start() - 15)  # 上下文起点：命中位置前 15 字符（防越界取 0）
            hits.append({"pattern": r["pattern"], "message": r["message"],  # 记录命中的规则与说明
                         "snippet": text[s:m.end() + 15].replace("\n", " ")})  # 命中上下文片段（换行替换为空格便于单行展示）
    return hits  # 空列表表示未命中任何红线


# ═════════════════════════════════════════════════════════════════
# check_action：校验 Reviewer 生成的进化动作（闭环的落盘前最后关卡）
# create 校验全文；patch 只校验新增/改写文本（纯删除放行）
# ═════════════════════════════════════════════════════════════════
def check_action(action: dict) -> list[dict]:
    """校验 Reviewer 生成的单个动作：
      create → 校验 content
      patch  → 校验 new_text（新增语义部分），并取其与 old_text 的增量
    返回违规列表，空列表表示通过。"""
    if action.get("action") == "create":  # 新建 Skill：整段 content 都是新增文本
        return check_text(action.get("content", ""))  # 对全文做红线扫描
    if action.get("action") == "patch":  # 修补 Skill：只关心改动引入的新语义
        new_text = action.get("new_text", "")  # 取补丁后的新文本
        # 若 new_text 是 old_text 的子集（纯删除），无需检查
        if new_text in action.get("old_text", ""):  # 纯删除不会引入危险表述 → 放行
            return []
        return check_text(new_text)  # 对新增/改写内容做红线扫描
    return []  # 未知动作类型不在此拦截（由上层 schema 校验把关）
