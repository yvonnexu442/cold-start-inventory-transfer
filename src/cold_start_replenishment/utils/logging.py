import json
import logging
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {"level": record.levelname, "logger": record.name, "message": record.getMessage()}
        )


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def log_frame_summary(logger: logging.Logger, name: str, frame: Any) -> None:
    missing = int(frame.isna().sum().sum())
    logger.info(
        "frame=%s rows=%d columns=%d missing=%d", name, len(frame), len(frame.columns), missing
    )
