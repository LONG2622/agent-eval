@echo off
:: agent.bat - Wrapper for agent-eval CLI
:: Usage: agent run "your task" / agent eval dataset.jsonl / agent list / etc.

set PYTHONPATH=%~dp0src;%PYTHONPATH%
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
python -m agent_eval %*
