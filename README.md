# 🧪 Agent Eval — Agent 运行与评估系统

> **Run agents, record full execution traces, and evaluate them systematically across 5 dimensions.**

一个开源的 Agent 评估框架：运行 Agent → 记录完整执行轨迹 → 从**任务成功率、工具调用、回答质量、延迟、Token Cost** 5 大维度系统评估。

支持 **LLM-as-Judge 语义评分**、**A/B 对比测试**、**Token 效率分析**、**回归测试基线**、**SQLite 持久化**、**HTML 可视化报告**、**CI/CD 流水线**、**Docker 一键部署**。

---

## ✨ 特性

### 核心引擎

| 模块 | 能力 |
|------|------|
| **🧠 LLM Gateway** | OpenAI 兼容 API、自动 Token 计量（tiktoken）、费用计算、指数退避重试、消息归一化、多模型 Profile 管理 |
| **🛠️ Tool Registry** | `@tool` 装饰器一键注册、Pydantic 参数校验、4 个内置工具（calculator / web_search / get_time / read_file） |
| **⚡ Agent Runtime** | `BaseAgent` 抽象 + 可插拔注册中心、ReAct Agent（Function Calling + Scratchpad 双模式自动切换） |
| **📜 Trace Recorder** | 步骤级 Span 全链路记录（LLM / Tool / Agent Step）、自动 Callback 挂载、防重复注册 |

### 评估引擎

| 模块 | 能力 |
|------|------|
| **📊 Evaluation Engine** | 5 大维度 × 20+ 细粒度指标、规则 + 关键词 + 启发式评估、批量聚合统计、**阈值全部 YAML 配置化** |
| **⚖️ LLM-as-Judge** | 5 维度语义评分（正确性 / 相关性 / 完整性 / 无害性 / 可读性）、Judge 失败自动降级到关键词评估 |
| **🔬 A/B Testing** | 双 Agent 同数据集对比、配对 t-test 统计显著性检验、差异分析报告 |
| **📉 Token 效率分析** | 冗余 Prompt 检测、上下文窗口占用率、chars/token 效率、系统提示膨胀比、自动诊断 flag（high_redundancy / context_near_limit / low_efficiency / empty_completion） |
| **🏷️ 回归测试基线** | 保存批量评估为 baseline、后续运行自动 diff、维度级 regression / improvement 检测、verdict 告警 |
| **📈 Batch Execution** | 并发执行（`--workers N`）、失败重试（指数退避）、断点续跑（checkpoint）、速率限制 |
| **🔴 Error Classifier** | 11 类错误自动分类（Timeout / RateLimit / Auth / Network / Internal / MaxSteps / Tool / ...）、错误汇总统计 |

### 存储与报告

| 模块 | 能力 |
|------|------|
| **💾 JSONL Storage** | runs / traces / evaluations 三类文件持久化（零依赖）、线程安全写入 |
| **🗄️ SQLite Storage** | 4 张关系表（tasks / runs / spans / evaluations）、CRUD、聚合查询、JSONL → SQLite 一键迁移 |
| **💻 Terminal Report** | Rich 时间线树 + KPI 卡片 + 维度汇总表 + Top 失败/成功案例 |
| **📄 HTML Report** | 3 种报告类型（单任务详情 / 批量汇总 / A/B 对比）、Chart.js 图表（雷达图 / 柱状图 / 饼图）、自包含 HTML |
| **🗂️ Human Annotation** | 人工评分（1–5 分）+ 10 类标签 + 自由评论、标注 CRUD、Ground Truth 积累 |
| **📊 Annotation vs Auto-Eval** | Pearson 相关系数、MAE/RMSE、散点图、标签分组分析、Top 差异案例 |

### Web 应用（FastAPI Server）

| 页面 / API | 能力 |
|-----------|------|
| **🧩 REST API** | 26 个端点：健康检查 / 配置 / 模型列表 / Runs CRUD / 批量评估 / A/B 对比 / Trace 回放 / 标注 CRUD / Judge / 错误分析 / 对比报告 |
| **📊 Web Dashboard** | ECharts KPI 卡片、成功率与 Token/延迟趋势、Runs 表格（排序 + 搜索 + 跳转 Trace/Annotate） |
| **🔍 Trace 回放** | 按 Step 分组的结构化时间线（Thought / LLM Call / Tool Call / Agent Step）、输入输出与 Token 详情、评估分数 |
| **✏️ 人工标注** | 1–5 分评分、10 类多选标签、评论、历史标注列表与删除、Ground Truth 导出入口 |
| **💬 交互式聊天** | Web 端 Chatbox 直接对话 Agent、可配置模型 / Steps / Temperature、实时推理步骤展示、自动评估分数、一键跳转 Trace / Annotate |
| **🔴 错误分析** | 11 类错误分类统计、错误详情表格、错误类型可视化图表 |
| **📊 标注对比** | 人工标注 vs 自动评估分数对比、相关性分析、散点图 / 雷达图 / 差异分布、Top 差异案例 |

### CLI（16 条命令）

```
agent run                  运行单个任务
agent eval                 批量评估数据集
agent report               查看运行报告（终端 / HTML）
agent list                 列出历史运行记录
agent evaluate             重新评估指定运行
agent compare              A/B 对比测试
agent migrate              JSONL → SQLite 数据迁移
agent config               查看 / 修改配置
agent judge                对已有运行执行 LLM-as-Judge
agent serve                启动 FastAPI Web 服务（Dashboard / Chat / Trace / Annotate）
agent errors               错误分类与统计
agent compare-annotations  标注 vs 自动评估对比报告
agent token-analysis       Token 效率分析（冗余 / 上下文占用 / 诊断 flag）
agent baseline-save        保存当前评估批次为回归基线
agent baseline-compare     当前运行 vs 基线对比（回归检测）
agent baseline-list        列出所有已保存基线
```

### 工程化

| 模块 | 能力 |
|------|------|
| **🧪 单元测试** | 361 个测试用例、覆盖率 84.6%（80% 门槛强制）、覆盖 20 个模块（trace / storage / tools / agent / runner / evaluation / judge / baseline / server / cli / reports / config） |
| **🔄 CI/CD** | GitHub Actions：ruff lint → 3 版本 Python 矩阵测试（3.10/3.11/3.12）→ Docker 构建，覆盖率门槛 80% |
| **🧹 代码规范** | ruff（E/F/W/I/B/UP 规则集）0 告警、类型注解、异常窄化（0 处裸 `except Exception`） |
| **🐳 Docker 部署** | 多阶段构建、非 root 用户、数据卷持久化、健康检查、日志轮转 |
| **📝 结构化日志** | 根/子 logger 分层（`--verbose` / `--json-logs` / `--log-file` 全局生效）、JSON 格式输出、文件日志 |
| **🔧 配置系统** | YAML + `.env` 环境变量替换、多模型 Profile、评估阈值配置化、热重载 |

---

## 🚀 快速开始

### 1. 安装依赖

```bash
cd Agent
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
pip install -e .
```

> **Windows 用户**：如果 `pip install -e .` 权限不足，可直接使用项目根目录的 `agent.bat` 脚本：
> ```bash
> .\agent.bat run "Calculate sqrt(144) + 5^2"
> ```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入你的 API 信息（3 个模型可选，默认用天大）：

```env
LLM_DEFAULT_MODEL=tju-llm

# 天津大学 LLM（中文友好，支持 Function Calling）
TJU_API_KEY=your-tju-key
TJU_BASE_URL=https://ai.tju.edu.cn/api/v3/
TJU_MODEL_NAME=tju-llm

# NVIDIA DeepSeek V4 Pro（强推理 + 代码 + 中文，支持 thinking）
NVIDIA_LLAMA_API_KEY=your-nvidia-key
NVIDIA_LLAMA_BASE_URL=https://integrate.api.nvidia.com/v1/
NVIDIA_LLAMA_MODEL_NAME=deepseek-ai/deepseek-v4-pro-0813

# NVIDIA Moonshot Kimi K3（多模态 + 长上下文 + 中文推理，不支持 FC，自动降级 Scratchpad）
NVIDIA_QWEN_API_KEY=your-nvidia-key
NVIDIA_QWEN_BASE_URL=https://integrate.api.nvidia.com/v1/
NVIDIA_QWEN_MODEL_NAME=moonshotai/kimi-k3
```

> 切换默认模型只需改 `LLM_DEFAULT_MODEL` 为上述任一模型名即可。

### 3. 运行单任务

```bash
agent run "Calculate sqrt(144) + 5^2 step by step" -e "37"
```

输出包含：
- 完整执行时间线（Thought → LLM Call → Tool Call → Observation → Final Answer）
- 5 维度评估表（成功率 / 工具调用 / 回答质量 / 延迟 / Token Cost）
- KPI 卡片（延迟 / Token / Cost / Steps）

### 4. 批量评估

```bash
# 串行评估
agent eval examples/sample_tasks.jsonl -n 5

# 并发评估（3 线程 + 自动重试 + HTML 报告）
agent eval examples/sample_tasks.jsonl -n 10 -w 3 --retries 2 --report-html
```

### 5. 查看报告

```bash
# 终端报告
agent report <run-id>

# HTML 报告（浏览器打开）
agent report <run-id> --format html
```

### 6. A/B 对比测试

```bash
agent compare examples/sample_tasks.jsonl -n 5 --report-html
```

### 7. LLM-as-Judge 评估

```bash
# 对已有运行执行 Judge 评分
agent judge <run-id>

# 在批量评估中启用 Judge（需在 configs/default.yaml 中设置 judge.enabled: true）
agent eval examples/sample_tasks.jsonl -n 5
```

### 8. 数据迁移

```bash
# JSONL → SQLite
agent migrate outputs
```

### 9. 启动 Web 服务（Dashboard + Chat + Trace + Annotate + Errors + Compare）

```bash
# 启动 FastAPI 服务，默认监听 127.0.0.1:8000
agent serve --host 127.0.0.1 --port 8000
```

服务启动后浏览器访问以下页面：

| 页面 | 地址 | 说明 |
|------|------|------|
| **Dashboard** | http://127.0.0.1:8000/ | KPI 卡片 + 运行趋势图 + 所有 Runs 列表（🔍 Trace / ✏️ Annotate 快捷链接） |
| **💬 交互式聊天** | http://127.0.0.1:8000/chat | 直接在网页端与 Agent 对话，配置模型参数，查看推理步骤、评估分数、一键标注 |
| **🔍 Trace 回放** | http://127.0.0.1:8000/trace/{run_id} | 结构化回放 Agent 的每一步 Thought / LLM Call / Tool Call |
| **✏️ 人工标注** | http://127.0.0.1:8000/annotate/{run_id} | 对一次 run 打分、加标签、写评语，用于生成 Ground Truth 改进评估模型 |
| **🔴 错误分析** | http://127.0.0.1:8000/errors | 失败运行的错误分类统计、错误详情表格、可视化图表 |
| **📊 标注对比** | http://127.0.0.1:8000/compare | 人工标注 vs 自动评估分数对比，含散点图、雷达图、差异分布等 |
| **REST API Docs** | http://127.0.0.1:8000/docs | Swagger 交互式 API 文档（26 个端点） |

### 10. 错误分类

```bash
# 查看所有失败运行的错误分类汇总
agent errors

# 限制显示的错误数量
agent errors --limit 5
```

### 11. 标注 vs 自动评估对比

```bash
# 运行对比分析（在终端显示结果）
agent compare-annotations

# 保存对比报告为 JSON 文件
agent compare-annotations --save
```

对比内容包含：
- 🔗 **Pearson 相关系数**：衡量人工评分与自动评估的一致性
- 📊 **MAE / RMSE**：量化两者的偏差程度
- 📈 **散点图**：每个运行点的人工 vs 自动分数分布
- 🏷️ **标签分组**：按人工标签分组的差异分析
- ⚠️ **Top 差异案例**：列出人工与自动评估分数差异最大的运行

### 12. Token 效率分析

```bash
# 批量分析所有运行：冗余 Prompt、上下文占用、chars/token 效率
agent token-analysis

# 只分析单个运行
agent token-analysis --run-id <run-id>
```

输出包含：
- 📋 每个 run 的 Prompt/Completion Token、冗余比例、上下文占用率、系统提示膨胀比
- 🚩 自动诊断 flag：`high_redundancy`（冗余 >25%）/ `context_near_limit`（占用 >80%）/ `low_efficiency` / `empty_completion`
- 🏆 Top 冗余运行排行 + 优化建议（如"考虑对话历史摘要压缩"）

### 13. 回归测试基线

```bash
# 1. 把当前评估批次保存为基线
agent baseline-save --name "v1.0 release"

# 2. （改进 Agent / 换模型 / 调 Prompt 之后）重新批量评估
agent eval examples/sample_tasks.jsonl -n 10

# 3. 与基线对比，自动检测性能退化
agent baseline-compare --baseline-id baseline_1787305000

# 4. 查看所有基线
agent baseline-list
```

对比输出：
- 📊 **Overall Delta**：当前 vs 基线的成功率差异
- 🔴 **Regressions**：退化的维度列表（如 `answer_quality 下降 12%`）
- 🟢 **Improvements**：提升的维度列表
- ⚖️ **Verdict**：一句话结论（"No regression" / "成功后退化了 8%"）

---

## 🐳 Docker 部署

### 快速部署

```bash
# 1. 克隆项目并进入目录
git clone <your-repo-url>
cd Agent

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，填入你的 API Key 和模型配置

# 3. 使用 Docker Compose 一键启动
docker compose up -d
```

服务将在 `http://localhost:8000` 启动，包含所有功能页面。

### 手动构建

```bash
# 构建镜像
docker build -t agent-eval:latest .

# 运行容器（映射端口 + 挂载数据卷）
docker run -d \
  --name agent-eval \
  -p 8000:8000 \
  -v agent_outputs:/app/outputs \
  -v agent_data:/app/data \
  --env-file .env \
  agent-eval:latest
```

### Docker Compose 配置说明

`docker-compose.yml` 提供了完整的生产级部署配置：

| 配置项 | 说明 |
|--------|------|
| **端口映射** | `8000:8000`（宿主机:容器） |
| **数据卷** | `agent_outputs`（运行输出）、`agent_data`（数据持久化） |
| **环境变量** | 通过 `.env` 文件注入 API 配置 |
| **健康检查** | 30 秒间隔检查 `/api/health` 端点 |
| **日志轮转** | json-file 驱动，10MB/文件，最多 3 个 |
| **自动重启** | `unless-stopped` 策略 |
| **安全配置** | 多阶段构建、非 root 用户运行 |

### 在 Docker 中使用 CLI

```bash
# 在运行中的容器内执行 CLI 命令
docker exec -it agent-eval agent run "Calculate 256 * 42"
docker exec -it agent-eval agent list
docker exec -it agent-eval agent compare-annotations
docker exec -it agent-eval agent errors
```

---

## 📁 项目结构

```
Agent/
├── .github/workflows/
│   └── ci.yml                    # GitHub Actions（ruff + 3 版本测试矩阵 + Docker build）
├── configs/
│   └── default.yaml              # 主配置（模型 / 定价 / Agent / Judge / 评估阈值）
├── examples/
│   └── sample_tasks.jsonl        # 示例任务（50 条，7 类别，3 档难度）
├── src/agent_eval/
│   ├── __main__.py               # python -m agent_eval 入口
│   ├── cli.py                    # CLI 入口（16 条命令）
│   ├── config.py                 # Pydantic 配置模型 + 环境变量替换 + 评估阈值配置
│   ├── logger.py                 # 结构化日志（根/子 logger 分层，JSON / 控制台）
│   │
│   ├── llm/                      # LLM 网关层
│   │   ├── gateway.py            # LLMGateway（计费 / 归一化 / Callback）
│   │   ├── messages.py           # Message 模型 + 序列化
│   │   ├── tokenizer.py          # Token 计数 + 费用计算
│   │   └── providers/
│   │       ├── base.py           # LLMProvider 抽象 + Callback
│   │       └── openai_provider.py # OpenAI 兼容实现（重试 / 多模型 Profile）
│   │
│   ├── tools/                    # 工具注册层
│   │   ├── registry.py           # ToolRegistry + @tool 装饰器 + Callback
│   │   └── builtin.py            # 4 个内置工具
│   │
│   ├── agent/                    # Agent 运行时
│   │   ├── base.py               # BaseAgent + AgentRegistry
│   │   └── react_agent.py        # ReAct Agent（Function Calling + Scratchpad 双模式）
│   │
│   ├── trace/                    # 轨迹记录层
│   │   ├── models.py             # RunRecord / Span / AnnotationRecord
│   │   ├── recorder.py           # TraceRecorder（三合一 Callback）
│   │   ├── storage.py            # JSONLStorage（线程安全）
│   │   └── sql_storage.py        # SQLiteStorage（4 表 + 聚合 + 迁移）
│   │
│   ├── task/                     # 任务执行层
│   │   └── runner.py             # TaskRunner（并发 / 重试 / 断点续跑 / 速率限制）
│   │
│   ├── evaluation/               # 评估引擎
│   │   ├── base.py               # BaseEvaluator + 数据模型
│   │   ├── builtin.py            # 5 大内置评估器（阈值从 YAML 读取）
│   │   ├── engine.py             # EvaluationEngine + BatchSummary
│   │   ├── llm_judge.py          # LLM-as-Judge 评估器
│   │   ├── ab_test.py            # A/B 测试引擎
│   │   ├── baseline.py           # 回归测试基线（save / compare / list）
│   │   ├── token_efficiency.py   # Token 效率分析器（冗余 / 上下文 / 诊断 flag）
│   │   └── error_classifier.py   # 11 类错误分类器
│   │
│   ├── report/                   # 报告层
│   │   ├── terminal_report.py    # Rich 终端报告
│   │   ├── html_report.py        # Chart.js HTML 报告
│   │   └── comparison_report.py  # 标注 vs 自动评估对比报告
│   │
│   └── server/                   # Web 服务层（原 app.py 拆分）
│       ├── app.py                # FastAPI 应用组装（CORS + 路由挂载）
│       ├── models.py             # 请求 Pydantic 模型
│       ├── state.py              # 共享状态（storage / engine）
│       ├── routes/
│       │   ├── api.py            # 18 个 REST API 端点
│       │   ├── pages.py          # 7 个 HTML 页面端点
│       │   └── websocket.py      # WebSocket 端点
│       └── templates/            # 6 个 HTML 模板（dashboard / chat / trace / annotate / errors / compare）
│
├── tests/                        # 单元测试（361 tests，覆盖率 84.6%）
│   ├── conftest.py               # 共享 fixtures
│   ├── test_trace_models.py      # Pydantic 数据模型
│   ├── test_storage.py           # JSONLStorage CRUD
│   ├── test_sql_storage.py       # SQLiteStorage + 迁移
│   ├── test_tools.py             # ToolRegistry + @tool
│   ├── test_react_agent.py       # ReAct Agent（双模式 mock 测试）
│   ├── test_recorder.py          # TraceRecorder 回调
│   ├── test_task_runner.py       # TaskRunner（并发 / 重试 / checkpoint）
│   ├── test_evaluation.py        # 5 个内置评估器
│   ├── test_engine.py            # EvaluationEngine
│   ├── test_llm_judge.py         # LLM-as-Judge（3 层解析降级路径）
│   ├── test_baseline.py          # 回归基线 save/compare
│   ├── test_token_efficiency.py  # Token 效率分析
│   ├── test_ab_test.py           # A/B 测试统计
│   ├── test_error_classifier.py  # 错误分类器
│   ├── test_comparison.py        # 对比报告
│   ├── test_reports.py           # 终端 + HTML 报告渲染
│   ├── test_llm_layer.py         # tokenizer / messages / gateway / provider
│   ├── test_server.py            # FastAPI 端点 + 页面 + WebSocket
│   ├── test_cli.py               # CLI 命令
│   └── test_config.py            # 配置加载
│
├── outputs/                       # 运行时输出（gitignore）
│   ├── traces/                    # 每条 Span 一行 JSONL
│   ├── runs/                      # Run 元信息
│   ├── evaluations/               # 评估结果 + 批量汇总
│   ├── annotations/               # 人工标注
│   └── baselines/                 # 回归测试基线
│
├── smoke_test.py                  # 离线冒烟测试（无需 API Key）
├── pyproject.toml                 # 打包 + pytest/coverage/ruff/black 配置
├── requirements.txt               # 依赖清单
├── Dockerfile                     # Docker 多阶段构建
├── docker-compose.yml            # Docker Compose 编排
├── .dockerignore                  # Docker 构建忽略规则
└── .env.example                   # 环境变量模板
```

---

## 📐 5 大评估维度

| 维度 | 评估方式 | 子指标 |
|------|---------|--------|
| **✅ 任务成功率** | Run 状态 + 关键词匹配 expected_output | PASS / KEYWORD_MATCH |
| **🛠️ 工具调用** | Span 聚合分析 | TOOL_CALL_COUNT、TOOL_SUCCESS_RATE、REDUNDANT_CALLS、综合分 |
| **🎯 回答质量** | 关键词正确率 + 完整度 + 拒绝检测；**LLM-as-Judge**（可选） | CORRECTNESS、COMPLETENESS、RELEVANCE、综合分 |
| **⏱️ 延迟** | 总耗时 + 每步平均 + LLM/工具占比 | TOTAL_LATENCY_MS、AVG_STEP_LATENCY_MS、LLM_PCT、TOOL_PCT |
| **💵 Token Cost** | Prompt/Completion 明细 + 总费用 + Token 效率 | PROMPT_TOKENS、COMPLETION_TOKENS、TOTAL_TOKENS、TOTAL_COST、EFFICIENCY |

### LLM-as-Judge 评分维度

启用后（`configs/default.yaml` → `judge.enabled: true`），回答质量维度额外获得 5 个语义评分（1-5 分）：

| Judge 维度 | 说明 |
|-----------|------|
| **Correctness** | 答案是否事实正确 |
| **Relevance** | 是否紧扣问题 |
| **Completeness** | 是否覆盖关键点 |
| **Harmlessness** | 是否有安全风险 |
| **Readability** | 语言是否流畅清晰 |

---

## 🔌 扩展点

### 注册自定义 Agent

```python
from agent_eval.agent import BaseAgent, register_agent
from agent_eval.trace import RunRecord, RunStatus

@register_agent
class MyCustomAgent(BaseAgent):
    agent_type = "my_agent"

    def run(self, task: str) -> tuple[str, RunRecord]:
        run = self._make_run(task)
        self.recorder.start_run(run)
        # 你的逻辑：调用 self.llm.chat() / self.tools.invoke()
        return output, self._finalize_run(run, output=output, status=RunStatus.SUCCESS)
```

### 注册自定义工具

```python
from agent_eval.tools import tool, ToolRegistry

@tool(name="my_tool")
def my_tool(query: str, top_k: int = 5) -> str:
    """Tool description.
    Args:
        query: Search query.
        top_k: Number of results.
    """
    return f"Results for {query} (top {top_k})"

registry = ToolRegistry()
registry.register(my_tool)
```

### 注册自定义评估器

```python
from agent_eval.evaluation import BaseEvaluator, EvalDimension, EvaluationResult

class MyEvaluator(BaseEvaluator):
    name = "my_eval"
    dimension = EvalDimension.ANSWER_QUALITY

    def evaluate(self, run, spans):
        return [EvaluationResult(
            run_id=run.run_id,
            evaluator=self.name,
            dimension=self.dimension,
            score=0.85,
            passed=True,
            details={"note": "custom logic"},
        )]
```

---

## 🧪 测试

### 单元测试

```bash
# 运行全量测试
python -m pytest tests/ -v

# 带覆盖率统计（需 pytest-cov，80% 门槛）
python -m pytest tests/ --cov=agent_eval --cov-report=term-missing --cov-fail-under=80
```

**361 个测试，覆盖率 84.6%**，覆盖 20 个模块：

| 测试文件 | 覆盖模块 | 测试数 |
|----------|----------|--------|
| test_trace_models.py | Pydantic 数据模型 | 16 |
| test_storage.py | JSONLStorage CRUD | 18 |
| test_sql_storage.py | SQLiteStorage + JSONL 迁移 | 19 |
| test_tools.py | @tool 装饰器 / ToolRegistry / 内置工具 | 30 |
| test_react_agent.py | ReAct Agent（Function Calling + Scratchpad 双模式） | 7 |
| test_recorder.py | TraceRecorder 回调记录 | 10 |
| test_task_runner.py | TaskRunner（并发 / 重试 / checkpoint 续跑） | 7 |
| test_evaluation.py | 5 个内置评估器 | 22 |
| test_engine.py | EvaluationEngine 批量评估 | 15 |
| test_llm_judge.py | LLM-as-Judge（3 层 JSON 解析降级路径） | 32 |
| test_baseline.py | 回归基线 save / compare / list | 14 |
| test_token_efficiency.py | Token 效率分析（冗余 / 上下文 / flag） | 14 |
| test_ab_test.py | A/B 测试统计（配对 t-test） | 15 |
| test_error_classifier.py | 错误分类器 + 汇总 | 19 |
| test_comparison.py | 标注对比报告 | 18 |
| test_reports.py | 终端 + HTML 报告渲染 | 16 |
| test_llm_layer.py | tokenizer / messages / gateway / provider | 19 |
| test_server.py | FastAPI 端点 + 页面 + WebSocket | 49 |
| test_cli.py | CLI 命令 | 9 |
| test_config.py | 配置加载 / 环境变量替换 | 12 |

### CI/CD

推送到 GitHub 后自动执行（[.github/workflows/ci.yml](.github/workflows/ci.yml)）：

1. **Lint** — ruff 全量检查（E/F/W/I/B/UP 规则集）
2. **Test** — Python 3.10 / 3.11 / 3.12 三版本矩阵 + 覆盖率门槛 80%
3. **Docker** — 镜像构建验证（GHA 缓存加速，不推送）

### 离线冒烟测试

无需 API Key，验证核心功能是否正常：

```bash
python smoke_test.py
```

覆盖 6 项：ToolRegistry / TraceRecorder / JSONLStorage / EvaluationEngine / TaskDataset / BatchAggregation

---

## 🔧 配置说明

主配置文件 `configs/default.yaml`，环境变量 `.env` 可覆盖：

```yaml
llm:
  default_model: tju-llm          # 默认模型
  temperature: 0.8
  max_tokens: 20000

agent:
  default_type: react
  max_steps: 10

evaluation:
  judge:
    enabled: false                 # 启用 LLM-as-Judge
    model: tju-llm                # 裁判模型
    temperature: 0.1

  # 评估器阈值全部可配置（无需改代码）
  quality:
    keyword_match_threshold: 0.6
    completeness_pass_threshold: 0.5
    relevance_pass_threshold: 0.6
  tool_usage:
    success_rate_threshold: 0.8
    success_weight: 0.6
    redundancy_weight: 0.4
  latency:
    total_budget_ms: 60000
    avg_step_budget_ms: 10000
  token_cost:
    max_total_tokens: 128000
    max_cost_usd: 1.0

pricing:
  tju-llm:
    prompt: 0.0                   # 按实际定价填写
    completion: 0.0
```

---

## 🗺️ 路线图

| 阶段 | 状态 | 核心交付 |
|------|------|---------|
| **Phase 1 — MVP** | ✅ 完成 | Agent Runtime + Trace Recorder + 5 维评估 + CLI + Terminal Report |
| **Phase 2 — 评估强化** | ✅ 完成 | LLM-as-Judge + A/B 对比 + SQLite + HTML 报告 + 并发批量 |
| **Phase 3 — Web 化** | ✅ 完成 | FastAPI REST API + Web Dashboard + Trace 回放 + 人工标注 + 交互式聊天 |
| **Phase 4 — 工程化** | ✅ 完成 | 错误分类器 + 标注对比报告 + Docker 部署 + 单元测试体系 + Bug 修复 |
| **Phase 5 — 质量与深度** | ✅ 完成 | app.py 模块化拆分 + 评估阈值配置化 + 异常窄化 + 日志分层修复 + CI/CD + 覆盖率 84.6% (361 tests) + Token 效率分析器 + 回归测试基线 |
| **Phase 6 — 体验与能力** | 🔜 计划中 | Chat 流式响应（SSE）+ WebSocket 实时推送 + 数据集管理页 + Dashboard 筛选搜索 + Plan-and-Execute Agent + 多模态支持 |

---

## 📝 License

MIT
