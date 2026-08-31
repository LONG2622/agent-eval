"""Structured logging setup for agent-eval."""

import json
import logging
import sys
from pathlib import Path
from typing import Any


class JsonFormatter(logging.Formatter):
    """JSON line formatter for machine consumption."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "extra"):
            log_entry.update(record.extra)  # type: ignore[attr-defined]
        return json.dumps(log_entry, ensure_ascii=False)


def setup_logger(
    name: str = "agent_eval",
    level: int | None = None,
    log_file: str | Path | None = None,
    json_output: bool = False,
) -> logging.Logger:
    """Configure and return a logger instance.

    Two modes:

    - **Root logger** (dotted-free name, e.g. ``agent_eval``): attaches console
      (+ optional file) handlers. The CLI's ``--verbose / --json-logs /
      --log-file`` options configure this one via ``setup_logger()``.
    - **Child logger** (dotted name, e.g. ``agent_eval.cli``): attaches no
      handlers and keeps ``propagate=True`` so records bubble up to the root
      logger and inherit its level/format/file settings.

    Args:
        name: Logger name.
        level: Logging level (root mode only; children inherit from parent).
        log_file: Optional file path to also write logs (root mode only).
        json_output: If True, output JSON lines (useful for pipelines).
    """
    logger = logging.getLogger(name)

    if "." in name:
        # Child logger: bubble up to the package root logger.
        logger.handlers.clear()
        logger.setLevel(logging.NOTSET)  # inherit effective level from parent
        logger.propagate = True
        return logger

    # Root logger: attach handlers.
    logger.setLevel(level if level is not None else logging.INFO)
    logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    if json_output:
        console_handler.setFormatter(JsonFormatter())
    else:
        fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
        console_handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(JsonFormatter())
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
