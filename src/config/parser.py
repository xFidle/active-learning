import logging
import tomllib
from pathlib import Path
from typing import TypeVar

from src.config.base import (
    DataclassInstance,
    get_all_registered,
    get_field_mappings,
    get_section_name,
    parse_config,
)

T = TypeVar("T", bound=DataclassInstance)


class ConfigParser:
    def __init__(self, config_path: Path | str = "config.toml"):
        self.config_path = Path(config_path)
        self._raw_config = self._load_toml()

    def _create_example_config(self) -> None:
        from dataclasses import fields

        config_sections = []

        for config_class in get_all_registered():
            section_name = get_section_name(config_class)
            field_mappings = get_field_mappings(config_class)
            default_instance = config_class()

            section_lines = [f"[{section_name}]"]

            for field in fields(config_class):  # type: ignore
                toml_key = field_mappings.get(field.name, field.name)
                value = getattr(default_instance, field.name)

                if isinstance(value, str):
                    formatted_value = f'"{value}"'
                elif isinstance(value, bool):
                    formatted_value = str(value).lower()
                elif isinstance(value, list):
                    formatted_value = str(value)
                elif isinstance(value, Path):
                    formatted_value = f'"{value}"'
                elif field.name == "level" and "Logger" in config_class.__name__:
                    formatted_value = f'"{logging.getLevelName(int(value))}"'
                elif value is None:
                    continue
                else:
                    formatted_value = str(value)

                section_lines.append(f"{toml_key} = {formatted_value}")

            config_sections.append("\n".join(section_lines))

        config_content = "\n\n".join(config_sections) + "\n"

        with open(self.config_path, "w") as f:
            f.write(config_content)

    def _load_toml(self) -> dict:
        if not self.config_path.exists():
            self._create_example_config()

        with open(self.config_path, "rb") as f:
            return tomllib.load(f)

    def get(self, config_class: type[T]) -> T:
        """
        Generic method to get any registered config.

        Example:
            parser = ConfigParser()
            logger_config = parser.get(LoggerConfig)
            img_config = parser.get(ImageProcessingConfig)
        """
        section_name = get_section_name(config_class)

        if section_name not in self._raw_config:
            raise ValueError(f"Missing [{section_name}] section in config file")

        section_data = self._raw_config[section_name]
        return parse_config(config_class, section_data)
