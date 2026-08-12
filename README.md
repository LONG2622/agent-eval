# 🧪 Agent Eval - Agent 运行与评估系统

> **Run agents, record full execution traces, and evaluate them systematically.**
> 从「任务成功率、工具调用、回答质量、延迟、Token Cost」5 大维度系统评估 Agent。

---

## ✨ 特性

| 模块 | MVP 能力 |
|------|----------|
| **🧠 LLM Gateway** | OpenAI 多模型支持、自动 Token 计量（tiktoken）、单次调用费用计算、指数退避重试 |
| **🛠️ Tool Registry** | `@tool` 装饰器一键注册、Pydantic 参数校验、沙箱执行、4 个内置示例工具 |
| **⚡ Agent Runtime** | `BaseAgent` 抽象 + 可插拔注册、ReAct Agent（Function Calling + Scratchpad 双模式） |
| **📜 Trace Recorder** | 步骤级 Span 记录（LLM/工具/思考/步骤）、自动挂到 LLM + 工具回调、JSONL 持久化 |
| **📊 Evaluation Engine** | 5 大维度 × 20+ 细粒度指标、规则 + 关键词 + 启发式评估器、批量汇总 |
| **💻 CLI + Terminal UI** | Rich 美化输出，完整 Trace 时间线回放、KPI 卡片、维度汇总表、Top 失败/成功案例 |

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

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 OPENAI_API_KEY，也可以改模型/价格/路径
```

### 3. 运行单任务 🔥

```bash
agent run "Calculate sqrt(144) + 5^2" --expected "sqrt(144)=12, 5^2=25, result=37"
```

会输出：
- 🧪 运行概览（Run ID、Agent、状态、输入）
- 📜 完整执行时间线树（每步：Thought → LLM Call → Tool Call → Observation）
- ✅ Final Answer
- 📊 指标卡片（延迟 / Token / Cost / Steps）
- 📋 评估结果表（5 维度 × 20+ 指标，含 Pass/Fail）

### 4. 批量评估数据集

```bash
agent eval examples/sample_tasks.jsonl --sample 5
```

会输出：
- 5 个 KPI 卡片：成功率 / 质量分 / 平均延迟 / 平均成本 / Runs&Tokens
- 维度汇总表：5 维度各自的 Pass Rate / Mean / Median / Min / Max
- ⚠️ Top 5 失败案例 + 🌟 Top 5 成功案例

### 5. 查看历史 Run

```bash
agent list                       # 所有最近 runs
agent report <run-id>            # 单个 run 的完整回放 + 评估
agent evaluate <run-id-1> ...    # 重新评估指定 run 或全部
```

---

## 📁 项目结构

```
Agent/
├── configs/default.yaml           # 主配置（带环境变量占位符）
├── examples/sample_tasks.jsonl    # 10 条示例任务
├── src/agent_eval/
│   ├── cli.py                     # CLI 入口 (typer)
│   ├── config.py                  # 配置加载 + 定价表
│   ├── logger.py                  # 结构化日志
│   ├── llm/                       # M3 LLM Gateway
│   │   ├── gateway.py
│   │   ├── messages.py
│   │   ├── tokenizer.py
│   │   └── providers/openai_provider.py
│   ├── tools/                     # M2 Tool Registry
│   │   ├── registry.py
│   │   └── builtin.py             # calculator/web_search/get_time/read_file
│   ├── agent/                     # M1 Agent Runtime
│   │   ├── base.py (BaseAgent + AgentRegistry)
│   │   └── react_agent.py
│   ├── trace/                     # M4 Trace Recorder
│   │   ├── models.py  (RunRecord, Span, TokenUsage)
│   │   ├── storage.py (JSONLStorage)
│   │   └── recorder.py (TraceRecorder 三合一 Callback)
│   ├── evaluation/                # M6/M7 Evaluation Engine
│   │   ├── base.py (Evaluator 接口 + 数据模型)
│   │   ├── builtin.py (5 大内置评估器)
│   │   └── engine.py (批量执行 + 聚合)
│   ├── task/                      # M5 Task Runner
│   │   └── runner.py (TaskItem + TaskDataset + TaskRunner)
│   └── report/                    # M9 Terminal Report
│       └── terminal_report.py
└── outputs/                       # 运行时输出 (gitignore)
    ├── traces/   {run_id}.jsonl   - 每条 Span 一行
    ├── runs/     runs.jsonl       - 每次 Run 元信息
    └── evaluations/               - 单任务 eval + batch summary
```

---

## 📐 5 大评估维度（MVP）

| 维度 | 评估方式 | 子指标 |
|------|---------|--------|
| **✔ 任务成功率** | Run 状态 + 关键词匹配 expected_output | PASS、KEYWORD_MATCH |
| **🛠️ 工具调用** | Span 聚合分析 | TOOL_CALL_COUNT、TOOL_SUCCESS_RATE、REDUNDANT_CALLS、综合分 |
| **🎯 回答质量** | 长度完整度 + 拒绝检测 + 关键词正确率 | COMPLETENESS、RELEVANCE、CORRECTNESS、综合分 |
| **⏱ 延迟** | 总耗时 + 每步平均 + LLM/工具占比 | TOTAL_LATENCY_MS、AVG_STEP_LATENCY_MS |
| **💵 Token Cost** | Prompt/Completion 明细 + 总费用 + Token 效率 | PROMPT_TOKENS、COMPLETION_TOKENS、TOTAL_TOKENS、TOTAL_COST、效率分 |

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
        # ... your logic here, call self.llm.chat / self.tools.invoke
        return output, self._finalize_run(run, output=..., status=RunStatus.SUCCESS)
```

### 注册自定义工具

```python
from agent_eval.tools import tool, ToolRegistry

@tool(name="my_tool")
def my_tool(query: str, top_k: int = 5) -> str:
    """Tool description goes here.
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
        return [EvaluationResult(run_id=run.run_id, ...)]
```

---

## 🔜 阶段 2/3/4 路线图

- **阶段 2**：LLM-as-Judge 质量评估、A/B 对比报告、SQLite 存储、HTML 报告
- **阶段 3**：FastAPI REST API + React Dashboard、数据集管理 Web UI、人工标注页
- **阶段 4**：CI/CD 回归测试、Token 利用率深度分析、错误分类器、监控告警

---

## 📝 License

MIT
