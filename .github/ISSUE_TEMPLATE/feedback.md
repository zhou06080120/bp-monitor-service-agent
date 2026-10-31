name: badcase 反馈
about: Demo 回答错误 / 推脱转人工时提交，真实 badcase 将进入 Skill 自进化闭环
title: "[badcase] "
labels: feedback
body:
  - type: textarea
    id: question
    attributes:
      label: 你的提问
      description: 在 Demo 里输入的原始问题
      placeholder: 例如：袖带应该绑在左臂还是右臂？
    validations:
      required: true
  - type: textarea
    id: answer
    attributes:
      label: Agent 的实际回答
      description: 粘贴 Demo 页面回答原文（若显示"联系人工客服"也请如实粘贴）
    validations:
      required: true
  - type: textarea
    id: expected
    attributes:
      label: 期望的正确回答（可选）
      description: 你认为应该怎么答；不填则由维护者核实
    validations:
      required: false
