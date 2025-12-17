import logging
import tomllib
from pathlib import Path

from src.image_processing.config import ImageProcessingConfig
from src.utils.logger import LoggerConfig


class ConfigParser:
    def __init__(self, config_path: Path | str = "config.toml"):
        self.config_path = Path(config_path)
        self._raw_config = self._load_toml()

    def _create_example_config(self) -> None:
        default_logger = LoggerConfig()
        default_img_processing = ImageProcessingConfig()

        level_name = logging.getLevelName(default_logger.level)

        config_content = f"""[image_processing]
model = "{default_img_processing.model}"
data_dir = "{default_img_processing.data_dir}"
output_dir = "{default_img_processing.output_dir}"

[logging]
log_output = {default_logger.output}
log_level = "{level_name}"
log_format = "{default_logger.format_string}"
"""

        with open(self.config_path, "w") as f:
            f.write(config_content)

    def _load_toml(self) -> dict:
        if not self.config_path.exists():
            self._create_example_config()

        with open(self.config_path, "rb") as f:
            return tomllib.load(f)

    def get_logger_config(self) -> LoggerConfig:
        if "logging" not in self._raw_config:
            raise ValueError("Missing [logging] section in config file")

        logging_section = self._raw_config["logging"]

        return LoggerConfig(
            output=logging_section.get("log_output", ["stdout"]),
            level=LoggerConfig.parse_log_level(logging_section.get("log_level", "INFO")),
            format_string=logging_section.get("log_format", "%(levelname)s - %(message)s"),
            log_file=logging_section.get("log_file"),
        )

    def get_image_processing_config(self) -> ImageProcessingConfig:
        if "image_processing" not in self._raw_config:
            raise ValueError("Missing [image_processing] section in config file")

        img_section = self._raw_config["image_processing"]

        return ImageProcessingConfig(
            model=img_section.get("model", "resnet50"),
            data_dir=Path(img_section.get("data_dir", "data/flowers/images")),
            output_dir=Path(img_section.get("output_dir", "data/flowers/processed")),
            force_download=img_section.get("force_download", False),
        )

    def get_all(self) -> tuple[LoggerConfig, ImageProcessingConfig]:
        return self.get_logger_config(), self.get_image_processing_config()
