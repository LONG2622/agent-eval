"""Quick test for Mistral Nemotron agent workflow."""
import sys
sys.path.insert(0, "src")

from agent_eval.config import load_config, reset_config
from agent_eval.llm import LLMGateway
from agent_eval.tools import ToolRegistry, tool
from agent_eval.trace import TraceRecorder
from agent_eval.agent import ReActAgent, AgentRunConfig

def calculator(expression: str) -> str:
    """Evaluate a math expression.
    
    Args:
        expression: Math expression to evaluate
    """
    return str(eval(expression))

def get_time() -> str:
    """Get current time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Reset config
reset_config()
config = load_config(force_reload=True)

# Create components
gateway = LLMGateway()
registry = ToolRegistry()
recorder = TraceRecorder()

# Register tools
calc_tool = tool(calculator)
time_tool = tool(get_time)
registry.register(calc_tool)
registry.register(time_tool)

# Create agent config
agent_config = AgentRunConfig(
    agent_name="test-mistral",
    agent_type="react",
    model="mistralai/mistral-nemotron",
    temperature=0.7,
    max_steps=10,
)

# Create agent
agent = ReActAgent(
    llm_gateway=gateway,
    tool_registry=registry,
    recorder=recorder,
    config=agent_config,
)

print("=" * 60)
print("Agent Configuration:")
print(f"  Model: {agent_config.model}")
print(f"  Use Function Calling: {agent._use_function_calling}")
print(f"  Tools: {[t.name for t in registry.list_tools()]}")
print("=" * 60)
print()

# Test 1: Simple calculation
print("Test 1: Simple calculation with tool")
print("-" * 40)
try:
    answer, run = agent.run("123 * 456 = ? Use calculator tool.")
    print(f"Answer: {answer}")
    print(f"Status: {run.status}")
    print(f"Steps: {run.total_steps}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 2: Chinese question
print("Test 2: Chinese question (no tool needed)")
print("-" * 40)
try:
    answer, run = agent.run("请简单介绍一下中国的首都北京")
    print(f"Answer: {answer[:200]}...")
    print(f"Status: {run.status}")
    print(f"Steps: {run.total_steps}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 3: Time query
print("Test 3: Get current time with tool")
print("-" * 40)
try:
    answer, run = agent.run("现在几点了？使用 get_time 工具查看当前时间。")
    print(f"Answer: {answer}")
    print(f"Status: {run.status}")
    print(f"Steps: {run.total_steps}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print()
print("All tests completed!")
