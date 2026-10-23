"""
后台回顾 Agent：分析近期对话，决定创建或修改哪些 Skill。

教学重点：
  1. 后台回顾 Agent 是独立的 LLM 调用，与主 Agent 解耦（Hermes 是异步 spawn）
  2. 回顾 Agent 拥有 policies.md（完整政策），主 Agent 没有——体现权限分层
  3. 回顾 Agent 输出结构化 JSON，由 SkillManager 执行 create/patch

使用方式：
  from background_reviewer import BackgroundReviewer
  reviewer = BackgroundReviewer("data/policies.md", sm)
  actions = reviewer.review(conversation_history)
  # actions 是 [{action, skill_name, ...}] 列表
"""

import os  # 读取 DEEPSEEK_API_KEY 等环境变量
import json  # 解析 Reviewer 输出的结构化 JSON 动作
import re  # 从回复中提取 JSON 串；抽取 Skill frontmatter 的 description
from openai import OpenAI  # OpenAI 兼容 SDK 调 DeepSeek
from pathlib import Path  # 读取完整政策文档 policies.md
from skill_manager import SkillManager  # 读取当前 Skill（供 Reviewer 参考，避免重复建）


# Reviewer 系统提示词（三引号字符串内部不可插注释）：定义"技能优化专家"角色、
# 三条核心原则（仅修复观察到的失败/最小改动/聚焦 1~2 类）、政策文档与已有 Skill
# 概览两个注入位，以及严格的 JSON 输出格式（注意双大括号 {{ }} 是转义的字面量大括号）
REVIEWER_SYSTEM = """你是康康血压计APP客服系统的"技能优化专家"。

以下给你的全部都是 Agent 最近一轮中**答错或推脱**的样本。你要做的是用最小改动
修复它们，让下次遇到同类问题能答对。

## 核心原则（严格遵守）

1. **仅修复观察到的失败**：只针对输入样本里出现的问题类型做改动，不要扩展到
   "政策里还有但样本里没出现"的场景
2. **最小改动优先**：
   - 能在已有 Skill 里追加或改一条分支解决的，不要新建 Skill
   - patch 的 old_text 只包含要改的那几行，不要把整段抄下来重写
3. **聚焦核心**：如果失败涉及多种类型，按失败条数从高到低，**只修复 1~2 类**
   —— 留出进化梯度，不要一次改完所有问题

你拥有完整政策文档，仅用于**核对 Agent 答错的具体数字/规则是否与政策一致**，
它是判定标准，不是 Agent 的知识补全大纲。

## 完整政策文档（判定标准）
{policies}

## 当前已有 Skill
{current_skills_summary}

## 输出格式
{{
  "analysis": "本轮失败 N 条，主要失败类型是 XXX",
  "actions": [
    {{"action": "create", "skill_name": "...", "reason": "修复哪条失败",
      "content": "完整SKILL.md（含frontmatter）"}},
    {{"action": "patch",  "skill_name": "...", "reason": "修复哪条失败",
      "old_text": "精确的原始文本", "new_text": "替换文本"}}
  ]
}}

只输出 JSON，不要有其他文字。如果发现失败数其实很少、没有清晰模式，可以只返回
1 条 action 甚至 0 条。"""

# Reviewer 用户消息模板：注入失败样本清单（数量+明细）与当前 Skill 全文
REVIEWER_USER = """## 本轮失败样本（共 {n} 条，都是 Agent 答错或推脱的）

{history_text}

## 当前 Skill 完整内容
{current_skills_full}

按核心原则给出最小必要的修复方案。"""


# ═════════════════════════════════════════════════════════════════
# BackgroundReviewer —— 失败复盘器（闭环的"进化大脑"）
# 输入：本轮评估出的失败样本 [{question, answer, fail_reason}]
# 输出：结构化动作列表（create/patch），经红线守卫校验后由
# SkillManager 落盘。拥有完整政策文档（权限高于主 Agent）
# ═════════════════════════════════════════════════════════════════
class BackgroundReviewer:
    def __init__(self, policies_path: str, skill_manager: SkillManager, model: str = "deepseek-chat"):
        self.policies = Path(policies_path).read_text(encoding="utf-8")  # 读入完整政策文档（Reviewer 的判定标准）
        self.skill_manager = skill_manager  # Skill 管理器（读当前技能；执行动作由外层负责）
        self.model = model  # 复盘模型
        self.last_analysis = ""   # UI 可读取最近一次回顾的分析文本
        # openai>=2 在 api_key=None 时构造即抛错：无 Key 的离线 Demo 也要能启动，
        # 用占位 Key 延迟到真正调 API 时才报错（get_components 预热不会崩）。
        api_key = os.getenv("DEEPSEEK_API_KEY") or "sk-offline-placeholder"  # 无 Key 时用占位符
        self.client = OpenAI(  # 构造 DeepSeek 客户端
            api_key=api_key,  # 认证密钥（可能是占位符）
            base_url="https://api.deepseek.com",  # API 入口
            timeout=float(os.getenv("LLM_TIMEOUT", "60")),  # 超时秒数
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),  # 重试次数
        )

    # ═══════════════════════════════════════════════════════════
    # review：复盘入口——组装失败样本 + 当前技能 → 调 LLM →
    # 解析出动作列表。空样本直接返回，不做任何 LLM 调用
    # ═══════════════════════════════════════════════════════════
    def review(self, failed_turns: list[dict]) -> list[dict]:
        """
        分析失败样本列表，返回最小必要的 Skill 操作。
        调用方应**仅传入本轮失败的条目**（每条形如 {question, answer, fail_reason}）；
        空列表直接返回 []，不做 LLM 调用。
        """
        if not failed_turns:  # 没有失败样本
            return []  # 无需复盘，直接返回空动作

        current_skills = self.skill_manager.load_all()  # 读当前全部技能（判断"改"还是"建"）
        skills_summary = "\n".join(  # 技能概览（名称+描述）：供系统提示词的 summary 注入位
            f"- {name}: {self._extract_description(content)}"
            for name, content in sorted(current_skills.items())  # 按名排序保证稳定
        ) or "（暂无已有Skill）"  # 空技能时的兜底文案
        skills_full = "\n\n---\n\n".join(  # 技能全文：供用户消息注入（patch 需要精确 old_text）
            f"### {name}\n{content}" for name, content in sorted(current_skills.items())
        ) or "（暂无已有Skill）"  # 空技能时的兜底文案

        system_msg = REVIEWER_SYSTEM.format(  # 组装系统消息：注入政策全文与技能概览
            policies=self.policies,  # 完整政策（Reviewer 特权，主 Agent 没有）
            current_skills_summary=skills_summary,  # 已有技能概览（防止重复创建）
        )
        user_msg = REVIEWER_USER.format(  # 组装用户消息：注入失败样本与技能全文
            n=len(failed_turns),  # 失败样本条数
            history_text=self._format_history(failed_turns),  # 失败样本明细（问题/回答/判错理由）
            current_skills_full=skills_full,  # 当前技能全文（patch 定位 old_text 用）
        )

        response = self.client.chat.completions.create(  # 调用 DeepSeek 生成修复方案
            model=self.model,  # 复盘模型
            messages=[
                {"role": "system", "content": system_msg},  # 角色+原则+政策+技能概览
                {"role": "user", "content": user_msg},  # 失败样本+技能全文
            ],
            temperature=0,  # 零温度：输出稳定可复现
            max_tokens=3000,  # 动作里含完整 SKILL.md 内容，放宽到 3000
        )
        return self._parse_actions(response.choices[0].message.content.strip())  # 解析回复为动作列表

    # ═══════════════════════════════════════════════════════════
    # _format_history：把失败样本渲染成 Reviewer 可读的编号清单
    # ═══════════════════════════════════════════════════════════
    def _format_history(self, turns: list[dict]) -> str:
        lines = []  # 收集渲染行
        for i, t in enumerate(turns, 1):  # 编号从 1 起
            lines.append(f"[{i}] 用户: {t['question']}")  # 用户问题行
            lines.append(f"    客服: {t['answer'][:200]}{'...' if len(t['answer']) > 200 else ''}")  # 客服回答行（截断 200 字）
            # 评估结果（如果调用方附带了失败原因）
            if t.get("fail_reason"):  # 有判错理由
                lines.append(f"    ✗ 判定：{t['fail_reason']}")  # 追加判错理由行（Reviewer 的关键线索）
        return "\n".join(lines)  # 拼接为多行文本

    def _extract_description(self, content: str) -> str:
        m = re.search(r"description:\s*(.+)", content)  # 从 frontmatter 提取 description 字段
        return m.group(1).strip() if m else "(无描述)"  # 无匹配则占位文案

    # ═══════════════════════════════════════════════════════════
    # _parse_actions：从 LLM 回复中提取 JSON 并解析出动作列表
    # 任何解析失败都安全降级为空列表（进化这轮跳过，不会污染技能库）
    # ═══════════════════════════════════════════════════════════
    def _parse_actions(self, raw: str) -> list[dict]:
        try:
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)  # 贪心提取首个 {...}（跨行；容忍模型多余废话）
            if not json_match:  # 连 JSON 形状都没有
                print(f"  [Reviewer] 无法提取 JSON，原始输出：{raw[:200]}")  # 打印原始输出前 200 字排查
                self.last_analysis = ""  # 清空分析文本
                return []  # 返回空动作
            data = json.loads(json_match.group())  # 解析 JSON
            self.last_analysis = data.get("analysis", "")  # 保存分析文本（UI 进化日志展示）
            print(f"  [Reviewer] 分析：{self.last_analysis[:100]}")  # 打印分析摘要
            return data.get("actions", [])  # 返回动作列表（可能为空=本轮不进化）
        except json.JSONDecodeError as e:  # JSON 损坏
            print(f"  [Reviewer] JSON 解析失败: {e}\n原始: {raw[:300]}")  # 打印错误与原文前 300 字
            self.last_analysis = ""  # 清空分析文本
            return []  # 安全返回空动作
