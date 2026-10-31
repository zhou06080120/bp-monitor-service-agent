# 公网部署与状态页手册

目标：把本项目变成「公开可试运行」的服务——在线 Demo + 第三方监控状态页 + 反馈闭环。
部署完成后的证据链：`README 徽章 → 状态页（UptimeRobot）→ Demo 站点 → /metrics 真实流量`，全部第三方可验证。

```
GitHub 仓库（含 render.yaml）
   │  push main 自动部署
   ▼
Render 容器（bp-monitor-service-agent，离线可起 / 配 Key 完整体验）
   ▲
   │  每 5 分钟心跳（兼防免费实例休眠）
UptimeRobot 免费监控 ──► 公开状态页 stats.uptimerobot.com/xxx
   │                        │
   ▼                        ▼
README 在线状态徽章      面试/简历直接引用的 uptime 链接
```

## 一、Render 部署（约 10 分钟，免费）

1. 注册 [render.com](https://render.com)（可用 GitHub 账号登录）
2. Dashboard → **New → Blueprint** → 选择本仓库（首次需授权 Render 访问 GitHub）
   - Render 会自动识别根目录 `render.yaml`，创建服务 `bp-monitor-service-agent`
3. 弹窗要求填两个环境变量（`sync: false`，只存在 Render 侧，不进仓库）：
   - `DEEPSEEK_API_KEY`：填你的 DeepSeek Key（完整对话体验）；**留空也能部署**，服务以离线形态启动（仅界面与检索，回答降级）
   - `BP_ADMIN_KEYS`：**必填**，一串随机长字符串（如 `openssl rand -hex 16` 生成），保护 `/reset` 与审批端点
4. 等待 build（首次约 3-5 分钟）→ 服务 URL 形如 `https://bp-monitor-service-agent.onrender.com`
5. 验证：
   - `curl https://<你的服务>.onrender.com/health` 返回正常
   - 浏览器打开根路径 `/` 出现三栏 Demo 页

**免费额度须知**：512MB RAM（离线形态足够，不要开 Milvus/Celery）；15 分钟无流量休眠（下一步的 UptimeRobot 心跳顺带解决）；每月 750 实例小时（单服务够用）。重启后 `outputs/` 复位到种子状态——对 Demo 反而是优点（任何人随时看到干净的初始形态与完整进化过程）。

**公开环境安全口径**（已写入 render.yaml，无需额外操作）：
- `RATE_LIMIT_RPM=20`：公网收紧限流
- `BP_APPROVAL_MODE=manual`：AI 产出的 Skill 变更不自动落盘，进待审批队列（`/approvals` 端点 + `BP_ADMIN_KEYS` 才能批准）——顺带把 P4 审批 gate 变成可现场演示的功能

## 二、UptimeRobot 状态页（约 10 分钟，免费）

1. 注册 [uptimerobot.com](https://uptimerobot.com)（免费版 50 个监控）
2. **Add New Monitor**：
   - Type: `HTTPS`
   - URL: `https://<你的服务>.onrender.com/health`
   - Monitoring Interval: `5 minutes`（>15 分钟休眠线，服务永不休眠）
3. 开启公开状态页：**My Settings → Public Status Pages → Add** → 勾选刚建的 monitor → 起名（如 `BP Agent Status`）→ 得到 `https://stats.uptimerobot.com/xxxxxxxx`
4. 拿徽章 URL：monitor 详情 → **Integrations → Badge**，或直接用 shields.io 格式：
   `https://img.shields.io/uptimerobot/ratio/m<monitorId>`

## 三、README 接线（把占位符换成真实地址）

`README.md` 顶部徽章区与「在线体验」一节中的占位符，按部署结果替换：

| 占位符 | 替换为 |
|---|---|
| `{YOUR-APP}.onrender.com` | Render 服务真实域名 |
| `{MONITOR_ID}` | UptimeRobot monitor 数字 ID（徽章 URL 里） |
| `{PAGE_ID}` | 状态页 URL 末段 |

## 四、国内访问评估与备选

`*.onrender.com` 在中国大陆访问时好时坏（取决于线路）。**部署完成后用手机蜂窝网络（关 Wi-Fi）打开 Demo 链接实测**：

- 能秒开 → 直接用，成本 0
- 偶尔慢/打不开 → 备选一：[Zeabur](https://zeabur.com)（对国内友好，容器直接部署本仓库，约 $5/月，GitHub 登录后导入仓库即可，其余配置同 `render.yaml`）
- 要稳定且正式 → 备选二：腾讯云/阿里云轻量应用服务器（约 40 元/月）+ 已备案域名，`docker run -p 80:8000` 即可（Dockerfile 已就绪）

## 五、把运行变成可引用的证据

- **uptime**：状态页本身即证据（第三方、不可自篡改），简历/README 直接挂链接
- **真实流量**：`GET /metrics`（Prometheus 文本，无需鉴权）：
  - `http_requests_total{...}` 按天/周截图留存，即"公开试运行 N 周、服务 X 次真实对话"
  - `rate_limit_rejected_total` 非零也是真实流量的旁证
- **反馈闭环**：README「在线体验」一节已挂 Issue 反馈入口（`.github/ISSUE_TEMPLATE/feedback.md`）。收到 badcase 后：
  1. 本地 `pytest` 全绿确认可复现
  2. 跑一轮 Nudge 进化修复
  3. Issue 里回复修复前后对比并 close
  ——这就攒出了「公开试运行收到 N 条真实反馈、修复 M 条」的独家叙事

## 六、常见问题

- **首次打开要等 30-60 秒？** 免费实例冷启动，UptimeRobot 心跳生效后不会再发生
- **配了 Key 还是回答降级？** 看 Render Dashboard → Logs，确认 `DEEPSEEK_API_KEY` 已保存且服务已重启
- **想重置实验状态？** `curl -H "X-API-Key: <BP_ADMIN_KEYS>" -X POST https://<域名>/reset`
- **不想公开 metrics？** 当前豁免设计用于"可审计"，如需关闭请自行在 `serve.py` 调整豁免列表（本方案默认保持公开）
