"""
LLM-as-a-Judge：轻量评判模型 + 结构化输出（JSON Mode）。

设计目标：
  替代"死板的关键词匹配"作为第二裁判。关键词规则仍保留为第一道硬闸
  （医疗安全相关：推脱、禁止词一票否决），Judge 只负责裁决「语义是否符合」
  的模糊地带 —— 例如同义改写（"3秒" vs "三秒"、"长按3秒" vs "按住3秒"）
  导致的 required 关键词缺失误判，由 Judge 依据语义判定翻转。

  结构化输出：response_format={"type": "json_object"}，返回 verdict/confidence/
  reason/evidence，避免自由文本解析。

配置（均可用环境变量覆盖，默认复用 DeepSeek）：
  JUDGE_API_BASE / JUDGE_API_KEY / JUDGE_MODEL
  —— 生产建议换成更轻量便宜的模型（如 deepseek-chat / qwen-turbo），
     因为 Judge 的任务是二元语义裁决，不需要强生成能力。
"""

import os  # 读取 Judge 专属配置环境变量（JUDGE_API_BASE/KEY/MODEL）
import json  # 解析 JSON Mode 返回的裁决对象
import re  # 从原始回复中兜底提取 JSON 字符串（防模型偶尔不守格式）
from openai import OpenAI  # OpenAI 兼容 SDK：Judge 模型同样走兼容协议调用

# Judge 系统提示词（三引号字符串内部不可插注释）：定义质检员角色、
# 判定四条标准（正确/完整/禁止项/推脱即 fail）与固定 JSON 输出格式
JUDGE_SYSTEM = """你是医疗器械客服质检员。给定【用户问题】、【标准要点】和【客服回答】，
判断客服回答在语义上是否**正确且完整**地覆盖了标准要点。

判定标准：
1. 「正确」：回答中的关键数值/规则与标准要点一致（允许同义改写，如"三秒"/"3秒"/"3 秒"，"22-32cm"/"22至32厘米"）；
2. 「完整」：标准要点中列出的 must 项不能缺失；
3. 「禁止项」：回答出现 forbidden 中任一语义（即使措辞不同）必须判 fail；
4. 回答含"联系人工"/"无法回答"等推脱语义时判 fail。

只输出 JSON，格式固定：
{"verdict": "pass" | "fail", "confidence": 0.0~1.0, "reason": "一句话结论", "evidence": ["命中的要点片段"]}"""

# Judge 用户消息模板：三个占位符分别填入问题、标准要点、客服回答
JUDGE_USER = """【用户问题】
{question}

【标准要点】
{ground_truth}

【客服回答】
{answer}"""


# ═════════════════════════════════════════════════════════════════
# LLMJudge —— 第二裁判（语义裁决层）
# 闭环位置：评估器第三层。规则硬闸与语义余弦判 fail 但疑似误杀时，
# 由 Judge 依据语义判 pass 翻案；推脱/禁止词永不翻案（安全优先）
# ═════════════════════════════════════════════════════════════════
class LLMJudge:
    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("JUDGE_MODEL", "deepseek-chat")  # Judge 模型（默认复用 deepseek-chat，可换便宜模型）
        self.call_count = 0  # 累计调用次数（评估成本统计用）
        # openai>=2 在 api_key=None 时构造即抛错：无 Key 环境（离线 Demo）
        # 也必须能创建 Judge 对象（available=False，Evaluator 自动降级语义模式）
        self.client = None  # 先置空：无 Key 环境保持对象可构造
        if os.getenv("JUDGE_API_KEY") or os.getenv("DEEPSEEK_API_KEY"):  # 有任一可用 Key 才建客户端
            self.client = OpenAI(  # 构造 Judge 专属客户端（与主 Agent 分开配置）
                api_key=os.getenv("JUDGE_API_KEY") or os.getenv("DEEPSEEK_API_KEY"),  # 优先 Judge 专属 Key，其次复用主 Key
                base_url=os.getenv("JUDGE_API_BASE", "https://api.deepseek.com"),  # API 入口（默认 DeepSeek）
                timeout=float(os.getenv("LLM_TIMEOUT", "60")),  # 请求超时秒数
                max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),  # 自动重试次数
            )

    @property
    def available(self) -> bool:
        return bool(os.getenv("JUDGE_API_KEY") or os.getenv("DEEPSEEK_API_KEY"))  # 有任一 Key 即视为可用（Evaluator 据此决定是否启用第三层）

    # ═══════════════════════════════════════════════════════════
    # judge：单次裁决。JSON Mode 保证结构化输出，temperature=0 保稳定
    # ═══════════════════════════════════════════════════════════
    def judge(self, question: str, ground_truth: dict, answer: str) -> dict:
        """返回 {"verdict": "pass"/"fail", "confidence": float, "reason": str, "evidence": list}
        LLM 不可用时抛异常（调用方负责降级到规则结果）。"""
        if self.client is None:  # 未配置任何 API Key
            raise RuntimeError("Judge 未配置 API Key（JUDGE_API_KEY / DEEPSEEK_API_KEY）")  # 显式报错，让调用方降级
        gt_text = self._format_ground_truth(ground_truth)  # 把 required/forbidden/note 渲染成 Judge 可读文本
        resp = self.client.chat.completions.create(  # 调用 Judge 模型
            model=self.model,  # 裁决模型
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},  # 系统提示：质检员角色 + 判定标准 + JSON 格式
                {"role": "user", "content": JUDGE_USER.format(  # 用户消息：填入问题/标准要点/回答三元组
                    question=question, ground_truth=gt_text, answer=answer[:800])},  # 回答截断 800 字控制成本
            ],
            temperature=0,  # 零温度：同一输入稳定输出
            max_tokens=300,  # 裁决结论很短，限 300 token 足够
            response_format={"type": "json_object"},   # JSON Mode：保证输出是合法 JSON
        )
        self.call_count += 1  # 调用计数 +1
        return self._parse(resp.choices[0].message.content.strip())  # 解析回复为标准裁决结构

    # ═══════════════════════════════════════════════════════════
    # vote：多数投票裁决。n 次 temperature>0 采样取多数，
    # 降低单次 LLM 裁决的随机性；返回值额外携带 votes 明细
    # ═══════════════════════════════════════════════════════════
    def vote(self, question: str, ground_truth: dict, answer: str,
             n: int = 3, temperature: float = 0.3) -> dict:
        """多数投票裁决：n 次采样（temperature>0 引入多样性），
        降低单次裁决的随机性；返回结构同 judge()，另附 votes 明细与投票分布。"""
        if self.client is None:  # 未配置 Key
            raise RuntimeError("Judge 未配置 API Key（JUDGE_API_KEY / DEEPSEEK_API_KEY）")  # 显式报错
        if n < 1:  # 投票次数非法
            n = 1  # 至少投一票
        gt_text = self._format_ground_truth(ground_truth)  # 渲染标准要点文本
        votes = []  # 收集 n 次独立裁决
        for _ in range(n):  # 采样 n 次
            resp = self.client.chat.completions.create(  # 每次独立调用 Judge 模型
                model=self.model,  # 裁决模型
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},  # 同一套系统提示
                    {"role": "user", "content": JUDGE_USER.format(  # 同一组三元组
                        question=question, ground_truth=gt_text, answer=answer[:800])},  # 回答同样截断
                ],
                temperature=temperature,  # 温度>0 引入采样多样性（这是投票有意义的前提）
                max_tokens=300,  # 限长
                response_format={"type": "json_object"},  # JSON Mode
            )
            votes.append(self._parse(resp.choices[0].message.content.strip()))  # 解析并收集该票
        self.call_count += n  # 调用计数加 n
        n_pass = sum(1 for v in votes if v["verdict"] == "pass")  # 统计 pass 票数
        n_fail = len(votes) - n_pass  # 统计 fail 票数
        majority = "pass" if n_pass > n_fail else "fail"  # 多数票决定最终裁决（平票时偏保守 fail）
        reasons = "; ".join({v["reason"] for v in votes if v["reason"]})[:200]  # 去重合并各票理由（截断 200 字）
        return {  # 返回投票聚合结果
            "verdict": majority,  # 多数裁决结论
            "confidence": max(n_pass, n_fail) / len(votes),   # 共识度作为置信度（如 2/3≈0.67）
            "reason": f"多数投票 {n_pass}/{len(votes)}: {reasons}",  # 票型分布 + 理由摘要
            "evidence": list({e for v in votes for e in v.get("evidence", [])})[:5],  # 各票证据去重合并（最多 5 条）
            "votes": [v["verdict"] for v in votes],  # 每票结论序列（审计用）
        }

    # ═══════════════════════════════════════════════════════════
    # _format_ground_truth：把评估集要点字典渲染成 Judge 可读的多行文本
    # ═══════════════════════════════════════════════════════════
    def _format_ground_truth(self, gt: dict) -> str:
        lines = []  # 逐行收集要点
        if gt.get("required"):  # 有必含关键词
            lines.append("must 必须出现：" + "、".join(gt["required"]))  # 渲染 must 行
        if gt.get("forbidden"):  # 有禁止关键词
            lines.append("forbidden 禁止出现：" + "、".join(gt["forbidden"]))  # 渲染 forbidden 行
        if gt.get("note"):  # 有补充说明
            lines.append("参考要点：" + gt["note"])  # 渲染 note 行
        return "\n".join(lines) or "（无）"  # 空要点兜底文案

    # ═══════════════════════════════════════════════════════════
    # _parse：把模型回复解析成标准裁决结构；解析失败返回 unknown
    # （unknown 视同无法裁决，调用方按规则结果处理，不会误翻案）
    # ═══════════════════════════════════════════════════════════
    def _parse(self, raw: str) -> dict:
        try:
            m = re.search(r"\{.*\}", raw, re.DOTALL)  # 兜底：从回复中提取首个 {...} JSON 串（跨行）
            data = json.loads(m.group() if m else raw)  # 提取到就用提取值，否则直接整段解析
            return {  # 归一化为标准结构（缺失字段给安全默认值）
                "verdict": "pass" if data.get("verdict", "").lower().startswith("pass") else "fail",  # 以 pass 开头才算 pass，其余一律 fail（保守）
                "confidence": float(data.get("confidence", 0.5)),  # 缺省置信度 0.5
                "reason": str(data.get("reason", ""))[:200],  # 理由截断 200 字
                "evidence": data.get("evidence", [])[:5],  # 证据最多留 5 条
            }
        except Exception as e:
            # JSON Mode 下几乎不会到这里；兜底视为无法裁决
            return {"verdict": "unknown", "confidence": 0.0, "reason": f"judge parse error: {e}", "evidence": []}  # unknown 不参与 pass/fail 统计
