"""
Skill 管理器：负责 Skill 文件的读取、创建、局部修改和版本历史追踪。

教学重点：
  1. Skill 文件是 Agent 的"可进化知识库"，与 RAG 原始文档不同
  2. patch 操作对应 Hermes 的 skill_manage(action="patch")
  3. 每次变更自动存版本快照，解决 Hermes 遗留问题#1（缺乏进化历史）

使用方式：
  from skill_manager import SkillManager
  sm = SkillManager("skills/")
"""

import re  # patch 时把 frontmatter 的 version 号 +1（正则替换）
import json  # 版本历史 JSON 文件的读写
from pathlib import Path  # 技能目录/版本目录/快照目录的路径操作
from datetime import datetime  # 每个版本条目记录变更时间


# ═════════════════════════════════════════════════════════════════
# SkillManager —— Skill 文件 CRUD + 版本历史 + 回滚（闭环的进化执行器）
# Reviewer 产出的 create/patch 动作经红线校验后由这里真正落盘；
# Reverse Eval Gate 判定退化时也由 rollback() 撤销最近一次变更
# ═════════════════════════════════════════════════════════════════
class SkillManager:
    """
    三重持久化设计：
      - skills/{name}/SKILL.md            活动版本，Agent 每次调用前读取
      - outputs/skill_versions/{name}_history.json  完整历史（含内容、时间、原因）
      - outputs/skill_snapshots/{name}_v{N}.md      每版独立 .md，方便 UI diff 或手工查看
    版本号 = 历史条目数（v1 = 初始版本）
    """

    def __init__(self, skills_dir: str, versions_dir: str = "outputs/skill_versions"):
        self.skills_dir = Path(skills_dir)  # 活动技能目录（skills/{name}/SKILL.md）
        self.versions_dir = Path(versions_dir)  # 版本历史 JSON 目录
        self.versions_dir.mkdir(parents=True, exist_ok=True)  # 确保历史目录存在
        self.snapshots_dir = Path(versions_dir).parent / "skill_snapshots"  # 独立快照 .md 目录（与历史目录同级）
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)  # 确保快照目录存在

    def load_all(self) -> dict[str, str]:
        """加载所有 Skill 文件，返回 {skill_name: content}"""
        skills = {}  # 结果字典：技能名 → SKILL.md 全文
        for skill_dir in self.skills_dir.iterdir():  # 遍历技能根目录下的每个子目录
            if skill_dir.is_dir():  # 只处理目录（每个技能一个目录）
                skill_file = skill_dir / "SKILL.md"  # 技能正文文件固定命名 SKILL.md
                if skill_file.exists():  # 文件存在才加载（空目录跳过）
                    skills[skill_dir.name] = skill_file.read_text(encoding="utf-8")  # 读入全文（UTF-8）
        return skills  # Agent 构建系统提示词的数据源

    def get(self, skill_name: str) -> str | None:
        skill_file = self.skills_dir / skill_name / "SKILL.md"  # 拼出目标技能文件路径
        if skill_file.exists():  # 存在则返回内容
            return skill_file.read_text(encoding="utf-8")  # 读取全文
        return None  # 不存在返回 None（调用方自行处理）

    def create(self, skill_name: str, content: str, reason: str = "") -> bool:
        """
        创建新 Skill。对应 Hermes 的 skill_manage(action="create")。
        同时保存版本快照供教学展示。
        """
        skill_dir = self.skills_dir / skill_name  # 新技能的专属目录
        skill_dir.mkdir(parents=True, exist_ok=True)  # 创建目录（已存在也不报错）
        skill_file = skill_dir / "SKILL.md"  # 技能正文文件路径
        if skill_file.exists():  # 同名技能已存在（不该发生，防重复创建）
            print(f"  [SkillManager] Skill '{skill_name}' 已存在，改用 patch")  # 提示应走 patch 语义
            return False  # 创建失败
        skill_file.write_text(content, encoding="utf-8")  # 写入技能正文（此刻新知识正式上线）
        self._save_version(skill_name, content, action="create", reason=reason)  # 落一条 v1 历史（含变更原因）
        print(f"  [SkillManager] ✓ 创建 Skill: {skill_name}")  # 打印创建成功日志（进化日志展示）
        return True  # 成功

    def patch(self, skill_name: str, old_text: str, new_text: str, reason: str = "") -> bool:
        """
        局部修改 Skill（字符串精确替换，仅替换第一处）。
        对应 Hermes 的 skill_manage(action="patch")。
        选择 replace 而非 diff/LLM-rewrite 是为了防止意外改动无关内容。
        """
        skill_file = self.skills_dir / skill_name / "SKILL.md"  # 目标技能文件路径
        if not skill_file.exists():  # 技能不存在无法修补
            print(f"  [SkillManager] ✗ Skill '{skill_name}' 不存在，无法 patch")  # 报错提示
            return False  # 失败
        content = skill_file.read_text(encoding="utf-8")  # 读入当前全文
        if old_text not in content:  # Reviewer 给的目标文本找不到（可能已被改过）
            print(f"  [SkillManager] ✗ 在 '{skill_name}' 中找不到目标文本")  # 报错提示
            return False  # 失败
        new_content = self._bump_version(content.replace(old_text, new_text, 1))  # 精确替换第一处命中，并把 frontmatter 版本号 +1
        skill_file.write_text(new_content, encoding="utf-8")  # 写回磁盘（修补即刻生效）
        self._save_version(skill_name, new_content, action="patch", reason=reason,
                           prev_content=content)  # 落历史：保留变更前全文（rollback 依据）
        print(f"  [SkillManager] ✓ 更新 Skill: {skill_name} (reason: {reason[:50]}...)")  # 打印更新日志（含原因摘要）
        return True  # 成功

    # ═══════════════════════════════════════════════════════════
    # rollback：Reverse Eval Gate 的执行入口
    # 按最近一次动作类型分派：create → 整套移除；patch → 恢复变更前全文
    # ═══════════════════════════════════════════════════════════
    def rollback(self, skill_name: str, reason: str = "rollback") -> bool:
        """撤销最近一次变更（按"最近一次动作类型"分派，而非按历史条数）：
        · 最近一次是 create → 该 Skill 为本次新建，整套移除（目录 + 历史 + 快照）
        · 最近一次是 patch  → 回退到 patch 前内容（含种子 Skill 首次被改动的场景）
        返回是否执行了回退。"""
        skill_dir = self.skills_dir / skill_name  # 技能目录路径
        skill_file = skill_dir / "SKILL.md"  # 技能正文路径
        history = self.get_version_history(skill_name)  # 读取该技能完整历史
        if not history or not skill_file.exists():  # 无历史或文件不存在 → 无可回滚
            print(f"  [SkillManager] ✗ 回退失败：Skill '{skill_name}' 不存在")  # 报错提示
            return False  # 失败

        history_file = self.versions_dir / f"{skill_name}_history.json"  # 历史文件路径（后面要改写/删除）
        last = history[-1]  # 最近一次变更条目

        if last.get("action") == "create":  # 最近一次是新建 → 本次进化是"凭空新增"
            # 本次新建的 Skill（create 必为历史首条，回滚 = 整体删掉）
            skill_file.unlink(missing_ok=True)  # 删除技能正文（missing_ok：不存在也不报错）
            try:  # 尝试删除技能目录
                skill_dir.rmdir()  # 目录非空会抛 OSError
            except OSError:  # 目录里还有其他文件
                pass  # 保留目录，只删正文也能让技能失效
            history_file.unlink(missing_ok=True)  # 删除历史 JSON（彻底抹掉该技能痕迹）
            for snap in self.snapshots_dir.glob(f"{skill_name}_v*.md"):  # 遍历该技能所有版本快照
                snap.unlink(missing_ok=True)  # 逐一删除
            print(f"  [SkillManager] ↩ 回滚（删除新建 Skill）: {skill_name}")  # 打印回滚日志
            return True  # 成功

        # patch 回退：恢复到 patch 前内容（prev_content 在 patch 时落盘）
        prev_content = last.get("prev_content")  # 优先取最近一条历史里存的变更前全文
        if prev_content is None:  # 旧格式历史可能没存 prev_content
            if len(history) >= 2:  # 有上一版可退
                prev_content = history[-2]["content"]  # 退到上一版内容
            else:  # 只有一条历史且无 prev_content → 无处可退
                print(f"  [SkillManager] ✗ 回退失败：'{skill_name}' 缺变更前快照")  # 报错提示
                return False  # 失败
        skill_file.write_text(prev_content, encoding="utf-8")  # 写回变更前全文（旧知识恢复生效）
        new_history = history[:-1]  # 历史弹出最近一条（版本数随之减一）
        self.versions_dir.mkdir(parents=True, exist_ok=True)  # 防御：历史目录可能被清理过
        history_file.write_text(json.dumps(new_history, ensure_ascii=False, indent=2),
                                encoding="utf-8")  # 改写历史文件（中文不转义、缩进 2）
        snap_file = self.snapshots_dir / f"{skill_name}_v{last.get('version', 1)}.md"  # 本次被回滚版本的快照文件
        snap_file.unlink(missing_ok=True)  # 删除该快照（目录里只留仍有效的版本）
        print(f"  [SkillManager] ↩ 回滚 Skill: {skill_name} → v{len(new_history)} "
              f"(reason: {reason[:60]})")  # 打印回滚摘要（目标版本 + 原因）
        return True  # 成功

    def get_version_history(self, skill_name: str) -> list[dict]:
        """返回某个 Skill 的所有历史版本（用于教学展示）"""
        history_file = self.versions_dir / f"{skill_name}_history.json"  # 历史文件路径
        if not history_file.exists():  # 从未变更过（种子技能或新建失败）
            return []  # 返回空列表
        return json.loads(history_file.read_text(encoding="utf-8"))  # 读取并解析历史条目列表

    def get_all_version_summaries(self) -> dict[str, list]:
        """返回所有 Skill 的版本历史摘要，格式：{skill_name: [{time, action, reason}]}"""
        summaries = {}  # 结果：技能名 → 摘要列表
        for skill_dir in self.skills_dir.iterdir():  # 遍历所有技能目录
            if skill_dir.is_dir():  # 只处理目录
                name = skill_dir.name  # 技能名
                history = self.get_version_history(name)  # 读该技能完整历史
                summaries[name] = [  # 压缩成摘要（去掉大体积的 content 字段）
                    {"time": h["time"], "action": h["action"], "reason": h.get("reason", "")}  # 只留时间/动作/原因
                    for h in history  # 遍历每条历史
                ]
        return summaries  # 供 UI 进化日志面板展示

    def get_active_versions(self) -> dict[str, int]:
        """返回当前所有 Skill 的版本号，用于 eval run 记录快照时标记"""
        result = {}  # 结果：技能名 → 当前版本号
        for skill_dir in self.skills_dir.iterdir():  # 遍历所有技能目录
            if skill_dir.is_dir():  # 只处理目录
                name = skill_dir.name  # 技能名
                history = self.get_version_history(name)  # 读历史
                result[name] = len(history)  # 版本号 = 历史条目数
        return result  # 评估结果文件用它标记"当时是什么版本组合"

    def _save_version(self, skill_name: str, content: str, action: str, reason: str,
                      prev_content: str | None = None):
        # 防御：版本/快照目录若被删除（如外部清理、reset 后部分恢复失败），
        # 写入前自动重建，避免 create/patch 静默落盘失败（曾导致进化动作全部丢失）
        self.versions_dir.mkdir(parents=True, exist_ok=True)  # 重建历史目录
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)  # 重建快照目录
        history_file = self.versions_dir / f"{skill_name}_history.json"  # 该技能的历史文件
        history = []  # 默认空历史
        if history_file.exists():  # 已有历史文件
            history = json.loads(history_file.read_text(encoding="utf-8"))  # 读出旧历史
        version_num = len(history) + 1  # 新版本号 = 旧条目数 + 1
        history.append({  # 追加一条变更记录
            "time": datetime.now().isoformat(),  # 变更时间
            "action": action,  # 动作类型：create / patch
            "reason": reason,  # 变更原因（来自 Reviewer 的分析结论）
            "version": version_num,  # 版本号
            "content": content,  # 变更后全文（当前活动内容）
            "prev_content": prev_content,  # 变更前全文（patch 时必填，rollback 用）
            "snapshot_file": f"skill_snapshots/{skill_name}_v{version_num}.md",  # 对应快照文件相对路径
        })
        history_file.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")  # 写回历史 JSON
        # 每个版本保存独立 .md 文件，UI 可直接按路径读取
        snapshot_path = self.snapshots_dir / f"{skill_name}_v{version_num}.md"  # 快照文件绝对路径
        snapshot_path.write_text(  # 写快照：头部两行 HTML 注释（版本/动作/时间/原因）+ 空行 + 全文
            f"<!-- {skill_name} v{version_num} | {action} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -->\n"
            f"<!-- reason: {reason[:100]} -->\n\n{content}",
            encoding="utf-8",
        )

    def _bump_version(self, content: str) -> str:
        """将 SKILL.md frontmatter 中的 version 字段加1"""
        def increment(m):  # re.sub 的替换函数：把匹配到的 version 数字 +1
            return f"version: {int(m.group(1)) + 1}"  # group(1) 是原数字，转 int 加一再格式化回去
        return re.sub(r"version:\s*(\d+)", increment, content)  # 全文替换所有 version 字段（正常只有 frontmatter 一处）
