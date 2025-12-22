import types
from dataclasses import fields
from typing import (
    Any,
    Callable,
    ClassVar,
    Protocol,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    runtime_checkable,
)


@runtime_checkable
class DataclassInstance(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Any]]


type FieldParser = Callable[..., Any]
T = TypeVar("T", bound=DataclassInstance)

T_Decorator = TypeVar("T_Decorator")

_REGISTRY: dict[type, tuple[str, dict[str, str], dict[str, FieldParser]]] = {}


def register_config(
    name: str,
    field_mappings: dict[str, str] | None = None,
    field_parsers: dict[str, FieldParser] | None = None,
):
    """
    Decorator to register a config dataclass.

    Args:
        section_name: TOML section name
        field_mappings: Map dataclass field names to TOML keys
        field_parsers: Custom parsers for specific fields

    Example:
        @register_config("logging", field_mappings={"output": "log_output"})
        @dataclass
        class LoggerConfig:
            output: list[str] = field(default_factory=lambda: ["stdout"])
    """

    def decorator(config_class: type[T_Decorator]) -> type[T_Decorator]:
        _REGISTRY[config_class] = (name, field_mappings or {}, field_parsers or {})
        return config_class

    return decorator


def parse_config(config_class: type[T], section_data: dict[str, Any]) -> T:
    if config_class not in _REGISTRY:
        raise ValueError(f"Config class {config_class.__name__} is not registered")

    _, field_mappings, field_parsers = _REGISTRY[config_class]
    kwargs = {}

    for field in fields(config_class):
        key = field_mappings.get(field.name, field.name)

        if key not in section_data:
            continue

        value = section_data[key]

        if field.name in field_parsers:
            value = field_parsers[field.name](value)
        else:
            field_type = field.type
            origin = get_origin(field.type)

            if origin is Union or origin is types.UnionType:
                type_args = [t for t in get_args(field_type) if t is not type(None)]
                if type_args:
                    field_type = type_args[0]

            if isinstance(field_type, type) and hasattr(field_type, "__dataclass_fields__"):
                value = parse_config(cast(type[DataclassInstance], field_type), value)
            elif field_type is bool and not isinstance(value, bool):
                if isinstance(value, str):
                    value = value.lower() in ("true", "1", "yes")
                else:
                    value = bool(value)
            elif isinstance(field_type, type) and not isinstance(value, field_type):
                value = field_type(value)

        kwargs[field.name] = value

    return config_class(**kwargs)


def get_all_registered() -> list[type]:
    return list(_REGISTRY.keys())


def get_section_name(config_class: type) -> str:
    if config_class not in _REGISTRY:
        raise ValueError(f"Config class {config_class.__name__} is not registered")
    return _REGISTRY[config_class][0]


def get_field_mappings(config_class: type) -> dict[str, str]:
    if config_class not in _REGISTRY:
        raise ValueError(f"Config class {config_class.__name__} is not registered")
    return _REGISTRY[config_class][1]
