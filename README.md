# 🧪 Agent Eval — Agent 运行与评估系统

> **Run agents, record full execution traces, and evaluate them systematically across 5 dimensions.**

一个开源的 Agent 评估框架：运行 Agent → 记录完整执行轨迹 → 从**任务成功率、工具调用、回答质量、延迟、Token Cost** 5 大维度系统评估。

支持 **LLM-as-Judge 语义评分**、**A/B 对比测试**、**SQLite 持久化**、**HTML 可视化报告**。

---

## ✨ 特性

### 核心引擎

| 模块 | 能力 |
|------|------|
| **🧠 LLM Gateway** | OpenAI 兼容 API、自动 Token 计量（tiktoken）、费用计算、指数退避重试、消息归一化 |
| **🛠️ Tool Registry** | `@tool` 装饰器一键注册、Pydantic 参数校验、4 个内置工具（calculator / web_search / get_time / read_file） |
| **⚡ Agent Runtime** | `BaseAgent` 抽象 + 可插拔注册中心、ReAct Agent（Function Calling + Scratchpad） |
| **📜 Trace Recorder** | 步骤级 Span 全链路记录（LLM / Tool / Agent Step）、自动 Callback 挂载、防重复注册 |

### 评估引擎

| 模块 | 能力 |
|------|------|
| **📊 Evaluation Engine** | 5 大维度 × 20+ 细粒度指标、规则 + 关键词 + 启发式评估、批量聚合统计 |
| **⚖️ LLM-as-Judge** | 5 维度语义评分（正确性 / 相关性 / 完整性 / 无害性 / 可读性）、Judge 失败自动降级到关键词评估 |
| **🔬 A/B Testing** | 双 Agent 同数据集对比、配对 t-test 统计显著性检验、差异分析报告 |
| **📈 Batch Execution** | 并发执行（`--workers N`）、失败重试（指数退避）、断点续跑（checkpoint）、速率限制 |

### 存储与报告

| 模块 | 能力 |
|------|------|
| **💾 JSONL Storage** | runs / traces / evaluations 三类文件持久化（零依赖） |
| **🗄️ SQLite Storage** | 4 张关系表（tasks / runs / spans / evaluations）、CRUD、聚合查询、JSONL → SQLite 一键迁移 |
| **💻 Terminal Report** | Rich 时间线树 + KPI 卡片 + 维度汇总表 + Top 失败/成功案例 |
| **📄 HTML Report** | 3 种报告类型（单任务详情 / 批量汇总 / A/B 对比）、Chart.js 图表（雷达图 / 柱状图 / 饼图）、自包含 HTML |

### CLI（9 条命令）

```
agent run        运行单个任务
agent eval       批量评估数据集
agent report     查看运行报告（终端 / HTML）
agent list       列出历史运行记录
agent evaluate   重新评估指定运行
agent compare    A/B 对比测试
agent migrate    JSONL → SQLite 数据迁移
agent config     查看 / 修改配置
agent judge      对已有运行执行 LLM-as-Judge
```

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

编辑 `.env`，填入你的 API 信息：

```env
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1      # 或你的兼容 API 地址
OPENAI_MODEL=gpt-4o-mini                        # 或你的模型名
```

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

---

## 📁 项目结构

```
Agent/
├── configs/
│   └── default.yaml               # 主配置（模型 / 定价 / Agent / Judge）
├── examples/
│   └── sample_tasks.jsonl         # 10 条示例任务
├── src/agent_eval/
│   ├── cli.py                     # CLI 入口（9 条命令）
│   ├── config.py                  # Pydantic 配置模型 + 环境加载
│   ├── logger.py                  # 结构化日志
│   │
│   ├── llm/                       # LLM 网关层
│   │   ├── gateway.py             # LLMGateway（重试 / 计费 / 归一化）
│   │   ├── messages.py            # Message 模型 + 序列化
│   │   ├── tokenizer.py           # Token 计数 + 费用计算
│   │   └── providers/
│   │       ├── base.py            # BaseProvider 抽象
│   │       └── openai_provider.py # OpenAI 兼容实现
│   │
│   ├── tools/                     # 工具注册层
│   │   ├── registry.py            # ToolRegistry + @tool 装饰器
│   │   └── builtin.py             # 4 个内置工具
│   │
│   ├── agent/                     # Agent 运行时
│   │   ├── base.py                # BaseAgent + AgentRegistry
│   │   └── react_agent.py         # ReAct Agent 实现
│   │
│   ├── trace/                     # 轨迹记录层
│   │   ├── models.py              # RunRecord / Span / EvalRecord
│   │   ├── recorder.py            # TraceRecorder（三合一 Callback）
│   │   ├── storage.py             # JSONLStorage
│   │   └── sql_storage.py         # SQLiteStorage（4 表 + 聚合 + 迁移）
│   │
│   ├── task/                      # 任务执行层
│   │   └── runner.py              # TaskRunner（并发 / 重试 / 断点续跑）
│   │
│   ├── evaluation/                # 评估引擎
│   │   ├── base.py                # BaseEvaluator + 数据模型
│   │   ├── builtin.py             # 5 大内置评估器
│   │   ├── engine.py              # EvaluationEngine + BatchSummary
│   │   ├── llm_judge.py           # LLM-as-Judge 评估器
│   │   └── ab_test.py             # A/B 测试引擎
│   │
│   └── report/                    # 报告层
│       ├── terminal_report.py     # Rich 终端报告
│       └── html_report.py         # Jinja2 + Chart.js HTML 报告
│
├── outputs/                       # 运行时输出（gitignore）
│   ├── traces/                    # 每条 Span 一行 JSONL
│   ├── runs/                      # Run 元信息
│   ├── evaluations/               # 评估结果 + 批量汇总
│   └── reports/                   # HTML 报告
│
├── smoke_test.py                  # 离线冒烟测试（无需 API Key）
├── pyproject.toml                 # 项目打包配置
├── requirements.txt               # 依赖清单
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

## 🧪 离线测试

无需 API Key，验证核心功能是否正常：

```bash
python smoke_test.py
```

覆盖 6 项：ToolRegistry / TraceRecorder / JSONLStorage / EvaluationEngine / TaskDataset / BatchAggregation

---

## � 配置说明

主配置文件 `configs/default.yaml`，环境变量 `.env` 可覆盖：

```yaml
llm:
  default_model: tju-llm          # 默认模型
  temperature: 0.7
  max_tokens: 2000

agent:
  default_type: react
  max_steps: 10

evaluation:
  judge:
    enabled: false                 # 启用 LLM-as-Judge
    model: tju-llm                # 裁判模型（可用更强模型）
    temperature: 0.1

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
| **Phase 3 — Web 化** | 🔜 计划中 | FastAPI REST API + Web Dashboard + Trace 回放 + 人工标注 |
| **Phase 4 — 工程化** | 🔜 计划中 | CI/CD 集成 + 错误分类器 + Token 利用率分析 + 监控告警 |

---

## 📝 License

MIT
