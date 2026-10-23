"""
医疗合规与输入安全 —— P3 合规加固基建

与姊妹项目（blood_pressure_agent / blodd_answer_piper）保持同款口径
（同步策略与 rate_limit.py / observability.py 一致），提供三个能力：

  1. MEDICAL_DISCLAIMER：统一医疗免责声明常量（API 响应字段与前端展示共用）
  2. validate_query()：用户输入校验与清洗（空输入 / 超长 / 控制字符剥离）
  3. check_injection()：常见 Prompt 注入模式检测（命中返回风险类别，未命中返回 None）

设计原则：
  - 零第三方依赖：纯标准库实现（不引入新的 pip 包）
  - 各服务按需取用：检索类服务只做输入校验（天然免疫注入），
    LLM 生成类服务叠加注入拦截；免责声明建议全响应统一附带
  - 误报优先级：注入模式仅覆盖高置信攻击语句，宁可漏报也不误伤正常医疗咨询
"""

import re  # 控制字符剥离与注入模式的正则

MEDICAL_DISCLAIMER = (  # 统一医疗免责声明（API 响应与前端页脚共用同一文案）
    "本回答仅供健康教育参考，不能替代执业医师诊疗意见；"
    "如有不适或异常读数，请及时就医。"
)

DEFAULT_MAX_INPUT = 500  # 用户输入长度上限（字符）；医疗咨询问题极少超过，超长多为滥用

# 控制字符（保留 \t \n \r 三个常用空白符）：剥离后防日志注入/协议 smuggling
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class ComplianceError(ValueError):
    """输入校验失败（空输入 / 超长）。code 供 API 层映射为不同的 400 detail。"""

    def __init__(self, code: str, message: str):
        super().__init__(message)  # message 直接作为用户可见的报错文案
        self.code = code           # invalid_input | input_too_long


def validate_query(text, max_length: int | None = None) -> str:
    """校验并清洗用户输入：None 拒绝 → 控制字符剥离 → strip → 空输入拒绝 → 超长拒绝。

    :param text: 用户原始输入
    :param max_length: 长度上限覆盖（默认 DEFAULT_MAX_INPUT）
    :return: 清洗后的安全文本
    :raises ComplianceError: 校验不通过（调用方转 400 响应）
    """
    max_length = max_length or DEFAULT_MAX_INPUT       # 未指定则用默认上限
    if text is None:                                   # 缺字段直接拒
        raise ComplianceError("invalid_input", "请输入您的问题")
    cleaned = _CONTROL_CHARS.sub("", str(text)).strip()  # 剥离控制字符并去首尾空白
    if not cleaned:                                    # 空串 / 纯空白 / 纯控制字符
        raise ComplianceError("invalid_input", "问题内容为空，请重新输入")
    if len(cleaned) > max_length:                      # 超长输入（多为滥用或粘贴错误）
        raise ComplianceError(
            "input_too_long",
            f"问题过长（{len(cleaned)}/{max_length} 字），请精简后重试")
    return cleaned                                     # 返回安全文本


# 常见 Prompt 注入模式（只收录高置信攻击语句，避免误伤正常咨询）：
#   - instruction_override：要求忽略既有指令（中/英文两式）
#   - role_hijack：要求扮演医务人员角色（绕过"不替代医生"的定位）
#   - system_probe：探测系统提示词 / 开发者模式
_INJECTION_PATTERNS = [
    ("instruction_override",
     re.compile(r"忽略.{0,8}(之前|以上|前面|所有).{0,8}(指令|提示|设定|规则)")),   # 中文式
    ("instruction_override",
     re.compile(r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?)",
                re.IGNORECASE)),                                                      # 英文式
    ("role_hijack",
     re.compile(r"(扮演|充当).{0,4}(医生|大夫|护士|药师)")),                        # 角色劫持
    ("system_probe",
     re.compile(r"(system\s*prompt|系统提示词|开发者模式|developer\s*mode)",
                re.IGNORECASE)),                                                      # 探测系统提示
]


def check_injection(text: str) -> str | None:
    """检测常见 Prompt 注入模式。

    :param text: 已经过 validate_query 清洗的文本（也可独立使用）
    :return: 命中返回风险类别字符串（instruction_override / role_hijack / system_probe），
             未命中返回 None
    """
    for category, pattern in _INJECTION_PATTERNS:  # 逐模式扫描
        if pattern.search(text):                   # 命中即返回（不做多类别聚合，够用）
            return category
    return None                                    # 全部未命中：安全
