from __future__ import annotations

import logging
from pathlib import Path
import sys
from typing import Any

from loguru import logger


class _InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame is not None and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def configure_server_logging(
    *,
    level: str = "INFO",
    log_file: str | Path | None = None,
    event_log_file: str | Path | None = None,
) -> None:
    """Configure loguru sinks and route standard logging through loguru."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        enqueue=True,
        backtrace=False,
        diagnose=False,
        filter=lambda record: not record["extra"].get("server_event", False),
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

    if log_file:
        path = Path(log_file).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            path,
            level=level,
            enqueue=True,
            backtrace=False,
            diagnose=False,
            rotation="100 MB",
            retention=10,
            filter=lambda record: not record["extra"].get("server_event", False),
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {process.id} | "
            "{thread.name} | {name}:{function}:{line} - {message}",
        )

    if event_log_file:
        path = Path(event_log_file).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            path,
            level="INFO",
            enqueue=True,
            backtrace=False,
            diagnose=False,
            serialize=True,
            filter=lambda record: record["extra"].get("server_event", False),
        )

    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    for name in ("websockets", "websockets.server"):
        logging.getLogger(name).setLevel(logging.INFO)


def log_server_event(event: str, **payload: Any) -> None:
    logger.bind(server_event=True, payload={"event": event, **payload}).info(event)
