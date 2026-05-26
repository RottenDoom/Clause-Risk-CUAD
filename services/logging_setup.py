"""
services/logging_setup.py

Loguru-based logging for the contract review pipeline.

Call configure_logging() once at startup. All other modules keep using the
standard library (logging.getLogger(__name__)) — the InterceptHandler routes
everything through loguru automatically.

Log level is read from the LOG_LEVEL env var (default: INFO).
Set LOG_LEVEL=DEBUG for Pinecone fetch counts, chunk scores, etc.
"""

import logging
import os
import sys
from pathlib import Path

from loguru import logger

LOG_FORMAT = (
    "<green>{time:YY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<level>{message}</level> | {extra}"
)

_configured = False


class _InterceptHandler(logging.Handler):
    """Route stdlib logging records through loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore[assignment]
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def configure_logging(
    level: str = "INFO",
    log_file: Path | None = None,
) -> None:
    """
    Configure loguru. Safe to call multiple times — only runs once.

    Args:
        level:    Default log level. Overridden by LOG_LEVEL env var if set.
        log_file: Path to the rotating log file.
                  Defaults to output/logs/contract_review.log.
    """
    global _configured
    if _configured:
        return
    _configured = True

    effective_level = os.environ.get("LOG_LEVEL", level).upper()

    # Remove loguru's default handler, then add ours
    logger.remove()
    logger.add(sys.stderr, format=LOG_FORMAT, level=effective_level, colorize=True)

    if log_file is None:
        log_dir = Path(__file__).parent.parent / "output" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "contract_review.log"

    logger.add(
        log_file,
        format=LOG_FORMAT,
        level=effective_level,
        rotation="10 MB",
        retention=5,
        colorize=False,
        encoding="utf-8",
    )

    # Intercept all stdlib logging (third-party libs + our modules) into loguru
    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)

    # Silence very noisy third-party loggers
    for name in (
        "httpx", "httpcore",
        "anthropic",
        "pinecone", "pinecone_plugin_interface",
        "sentence_transformers", "transformers", "tokenizers",
        "chromadb",
        "uvicorn.access",
        "urllib3", "filelock",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)

    logger.info("Logging configured level={} file={}", effective_level, log_file)
