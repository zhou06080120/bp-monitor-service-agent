"""
客服 Agent：使用当前 Skills 回答用户问题，内置 Nudge 计数机制。

教学重点：
  1. Agent 的唯一知识来源是 Skill 文件，不直接访问 policies.md
  2. Nudge 计数器：每 N 次调用后触发后台回顾（对应 Hermes _iters_since_skill）
  3. 每次回答前动态加载最新 Skills，保证进化后立即生效

使用方式：
  from agent import CustomerServiceAgent
  from skill_manager import SkillManager
  sm = SkillManager("skills/")
  agent = CustomerServiceAgent(sm, nudge_interval=20)
  answer = agent.answer("血压计怎么开机？需要按多久？")
"""

import os  # 读取环境变量：DEEPSEEK_API_KEY / LLM_TIMEOUT / LLM_MAX_RETRIES
from openai import OpenAI  # OpenAI SDK：DeepSeek 兼容 OpenAI 协议，复用官方客户端调用
from skill_manager import SkillManager  # Skill 文件管理器：load_all() 每次从磁盘读最新技能
from skill_store import SkillStore  # Skill 分块索引：技能多时提供混合检索（Dense+BM25+RRF 融合）
from observability import llm_call_timer  # LLM 调用统一埋点（llm_calls_total / llm_call_duration_seconds）


# 系统提示词总模板（三引号字符串内部不可插注释）：定义客服人设与
# "能答/不能答"两分支硬规则，{skills_section} 占位符由 _build_system_prompt 动态填充
SYSTEM_TEMPLATE = """你是康康血压计APP的智能客服助手。

你的所有知识来源于以下技能文档，严格基于文档内容回答，不要自行推断或编造政策。

## 回答规则（严格遵守）
- 【能回答】如果技能文档覆盖了用户问题：直接给出完整具体的答案（含具体天数/金额/
  工作日数等政策细节）。**不要在答案中加"建议联系人工客服"之类的推脱话**。
- 【不能回答】如果技能文档确实不覆盖：**仅回答一句** "需要联系人工客服"，
  不要编造答案，也不要列举可能的情况。

{skills_section}
"""

# 全量挂载模板：Skill 数量少时把所有技能原文整体塞进系统提示词
SKILLS_SECTION_TEMPLATE = """## 当前知识库（共{count}个技能）

{skills_content}
"""

# 动态挂载提示模板：仅注入检索命中的规则分块（带溯源小节）
MOUNTED_SECTION_TEMPLATE = """## 当前知识库（{total} 个技能中按相关性挂载 {count} 条规则）

{skills_content}

注意：以上只是与当前问题最相关的规则片段；未挂载的技能不参与本次回答。
"""

# 超过 4 个 Skill 或总字符超过预算 → 切换为"向量检索 + 动态按需挂载"
INLINE_SKILL_MAX = 4  # 全量内联挂载的技能数量上限（超过则改走检索挂载）
INLINE_CHAR_MAX  = 6000  # 全量内联挂载的总字符预算（防止 Prompt 膨胀拉高 Token 成本）


# ═════════════════════════════════════════════════════════════════
# CustomerServiceAgent —— 客服应答主体（闭环的知识消费端）
# 流程：加载最新 Skill → 少则全量内联/多则检索挂载 → 组装系统提示词 →
# 调 DeepSeek 生成回答；Nudge 计数器决定何时该触发后台失败复盘。
# ═════════════════════════════════════════════════════════════════
class CustomerServiceAgent:
    def __init__(
        self,
        skill_manager: SkillManager,  # Skill 文件管理器（每次回答前重载最新技能）
        nudge_interval: int = 20,  # 每 N 次回答触发一次后台复盘（Hermes 式 Nudge；0=禁用）
        model: str = "deepseek-chat",  # DeepSeek 对话模型名
        skill_store: SkillStore | None = None,  # 可选分块索引；None 则永远全量内联挂载
        mount_k: int = 8,  # 检索挂载时取 Top-K 个最相关规则分块
        mount_budget_chars: int = 4000,  # 检索挂载的字符预算（防 Prompt 过长）
    ):
        self.skill_manager = skill_manager  # 保存管理器引用，供每次 answer 时重读磁盘
        self.nudge_interval = nudge_interval  # Nudge 周期阈值
        self.model = model  # 模型名（构造后不再变）
        self.skill_store = skill_store  # 分块索引（可能为 None）
        self.mount_k = mount_k  # 检索 Top-K
        self.mount_budget_chars = mount_budget_chars  # 挂载字符预算
        self.last_mount: dict = {}   # 最近一次回答的挂载信息（供 UI/日志展示）
        self._iters_since_nudge = 0  # 距上次 Nudge 的累计回答次数（answer 一次 +1）
        # conversation_history 仅为后台回顾 Agent 保留观察样本；
        # 主 Agent 每次 answer() 都不把它传给 LLM（保证每题独立评估）。
        self.conversation_history: list[dict] = []  # 滚动观察窗口（最多 50 条问答快照）

        # openai>=2 在 api_key=None 时构造即抛错：无 Key 环境用占位 Key 延迟到
        # 真正调 API 时才报错（离线 Demo 只构建 Prompt/做检索不需要 Key）
        self.client = OpenAI(  # 构造 DeepSeek 客户端（OpenAI 兼容协议）
            api_key=os.getenv("DEEPSEEK_API_KEY") or "sk-offline-placeholder",  # 无 Key 环境用占位符延迟报错
            base_url="https://api.deepseek.com",  # DeepSeek API 入口地址
            timeout=float(os.getenv("LLM_TIMEOUT", "60")),  # 单次请求超时秒数（默认 60s）
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),  # SDK 层自动重试次数（默认 2 次）
        )

    # ═══════════════════════════════════════════════════════════
    # answer：单题问答入口。每题都重建系统提示（注入最新 Skill）、
    # 不携带历史消息 → 评估互不干扰；同时记账供 Nudge/后台复盘使用
    # ═══════════════════════════════════════════════════════════
    def answer(self, question: str) -> str:
        """
        回答单个问题。每次调用都会重新加载最新 Skills（保证 Nudge 后立即生效），
        且 messages 里只含系统提示 + 当前问题（不携带 conversation_history），
        这样每次评估互不干扰。
        """
        # 调 DeepSeek Chat 接口；temperature=0 贪心解码保证评估可复现；
        # llm_call_timer 统一埋点：LLM 调用成败计数 + 耗时观测（异常原样上抛）
        with llm_call_timer():
            response = self.client.chat.completions.create(
                model=self.model,  # 使用构造时指定的模型
                messages=[
                    {"role": "system", "content": self._build_system_prompt(question)},  # 系统提示：内含按相关性挂载的 Skill 知识
                    {"role": "user", "content": question},  # 用户消息：仅当前问题（不带历史，保证每题独立）
                ],
                temperature=0,  # 零温度采样：输出稳定，便于回归对比
                max_tokens=400,  # 限制回答长度，控制成本与延迟
            )
        answer_text = response.choices[0].message.content.strip()  # 取首个候选回答并去首尾空白

        # 对话历史只做滚动观察窗口，供后台回顾 Agent 使用
        self.conversation_history.append({  # 记录本轮问答快照（问题+答案+当时技能清单）
            "question": question,  # 用户原问题
            "answer": answer_text,  # Agent 本轮回答
            "skills_used": list(self.skill_manager.load_all().keys()),  # 回答时点的技能名列表（供复盘溯源）
        })
        if len(self.conversation_history) > 50:  # 窗口超过 50 条
            self.conversation_history = self.conversation_history[-50:]  # 只保留最近 50 条（滚动窗口）

        self._iters_since_nudge += 1  # Nudge 计数器累加（达到阈值后由外层触发复盘）
        return answer_text  # 返回纯文本回答给调用方（评估器/流水线）

    # ═══════════════════════════════════════════════════════════
    # should_trigger_nudge：判断是否到达后台复盘触发点（触发即归零计数器）
    # ═══════════════════════════════════════════════════════════
    def should_trigger_nudge(self) -> bool:
        """判断是否应该触发后台回顾。触发后计数器归零。"""
        if self.nudge_interval > 0 and self._iters_since_nudge >= self.nudge_interval:  # 启用了 Nudge 且计数已达周期
            self._iters_since_nudge = 0  # 归零计数器，开始下一轮观察
            return True  # 告知外层：该触发后台复盘了
        return False  # 未到周期或 Nudge 已禁用

    # ═══════════════════════════════════════════════════════════
    # reset_nudge_counter：外部已主动做 Skill 管理时手动清零，避免重复复盘
    # ═══════════════════════════════════════════════════════════
    def reset_nudge_counter(self):
        """Agent 主动调用 skill_manage 时手动归零，避免重复触发。"""
        self._iters_since_nudge = 0  # 计数器直接清零

    # ═══════════════════════════════════════════════════════════
    # _should_mount_dynamically：全量内联 vs 检索挂载的切换判据（Token 成本控制）
    # ═══════════════════════════════════════════════════════════
    def _should_mount_dynamically(self, skills: dict[str, str]) -> bool:
        """Token 成本控制：Skill 多或总字符超预算时启用检索挂载。"""
        return (self.skill_store is not None  # 前提：必须提供了分块索引才可能走检索
                and (len(skills) > INLINE_SKILL_MAX  # 技能数超过内联上限
                     or sum(len(c) for c in skills.values()) > INLINE_CHAR_MAX))  # 或技能总字符超过预算

    # ═══════════════════════════════════════════════════════════
    # _build_system_prompt：组装系统提示词——技能少则全量内联，
    # 多则混合检索 Top-K 分块动态挂载；挂载详情写入 last_mount 供审计展示
    # ═══════════════════════════════════════════════════════════
    def _build_system_prompt(self, question: str = "") -> str:
        skills = self.skill_manager.load_all()  # 每次重新读盘，拿到进化后的最新技能
        self.last_mount = {"mode": "inline", "mounted": len(skills)}  # 先默认记录为全量内联模式

        if not skills:  # 一个技能都没有（如 reset 之后）
            skills_section = "（暂无技能文档，请依据通用客服原则回答）"  # 兜底文案（会触发"联系人工"分支）
        elif self._should_mount_dynamically(skills):  # 技能多/超预算 → 走检索挂载
            hits = self.skill_store.retrieve(question, k=self.mount_k,  # 混合检索出 Top-K 相关规则分块
                                             budget_chars=self.mount_budget_chars)  # 同时受字符预算硬约束
            if not hits:  # 检索空转兜底：退化为全量挂载
                # 索引可能过期失效 → 先全量重建索引再重试一次（返回值仅为副作用，不使用）
                hits_ = self.skill_store.reindex_all(skills)  # noqa: F841
                hits = self.skill_store.retrieve(question, k=self.mount_k,  # 重建后再次检索
                                                 budget_chars=self.mount_budget_chars)
            # 每个命中分块格式化为带"技能名/小节名"溯源头的 Markdown 小节
            blocks = [f"### [{h['meta']['skill']} / {h['meta']['section']}]\n{h['text']}"
                      for h in hits]  # 遍历所有命中分块生成列表
            skills_section = MOUNTED_SECTION_TEMPLATE.format(  # 用动态挂载模板包裹分块集合
                total=len(skills), count=len(hits),  # 告知 LLM：总技能数 vs 实际挂载数
                skills_content="\n\n---\n\n".join(blocks))  # 分块之间用分隔线连接
            self.last_mount = {  # 记录检索挂载详情（UI 审计面板展示）
                "mode": "retrieval",  # 挂载模式：检索
                "mounted": len(hits),  # 实际挂载的分块数
                "total": len(skills),  # 技能总数
                "backend": self.skill_store.store.backend,  # 向量后端类型（local / milvus）
                "sections": [f"{h['meta']['skill']}/{h['meta']['section']}" for h in hits],  # 命中分块清单（溯源）
                "chars": sum(len(h["text"]) for h in hits),  # 本次挂载总字符数
            }
        else:  # 技能少且未超预算 → 全量内联挂载
            parts = []  # 收集每个技能的格式化文本
            for name, content in sorted(skills.items()):  # 按技能名排序，保证 Prompt 内容稳定
                parts.append(f"### 技能：{name}\n{content}")  # 单个技能格式：标题 + 全文
            skills_section = SKILLS_SECTION_TEMPLATE.format(  # 用全量挂载模板包裹
                count=len(skills),  # 技能总数
                skills_content="\n\n---\n\n".join(parts),  # 技能之间用分隔线连接
            )
        return SYSTEM_TEMPLATE.format(skills_section=skills_section)  # 知识库正文填入总模板并返回
