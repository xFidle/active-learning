import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from src.config.base import register_config

LogOutput = Literal["file", "stdout"]

level_mapping = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def parse_log_level(level_str: str) -> int:
    """Parse log level string to logging constant."""
    level_upper = level_str.upper()
    if level_upper not in level_mapping:
        raise ValueError(
            f"Invalid log level: {level_str}. Must be one of: {', '.join(level_mapping.keys())}"
        )
    return level_mapping[level_upper]


@register_config(
    section_name="logging",
    field_mappings={
        "output": "log_output",
        "level": "log_level",
        "format_string": "log_format",
    },
    field_parsers={"level": parse_log_level},
)
@dataclass
class LoggerConfig:
    level: int = logging.INFO
    output: list[LogOutput] = field(default_factory=lambda: ["stdout"])
    log_file: Path | str | None = None
    format_string: str = "%(levelname)s - %(message)s"
