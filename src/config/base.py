from typing import TYPE_CHECKING, Any, Callable, ClassVar, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from src.config.parser import ConfigParser


@runtime_checkable
class DataclassInstance(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Any]]


type FieldParser = Callable[[Any, "ConfigParser"], Any]
type FieldSerializer = Callable[[Any], Any]
T = TypeVar("T", bound=DataclassInstance)

T_Decorator = TypeVar("T_Decorator")

_REGISTRY: dict[
    type, tuple[str, dict[str, str], dict[str, FieldParser], dict[str, FieldSerializer]]
] = {}


def register_config(
    name: str,
    field_mappings: dict[str, str] | None = None,
    field_parsers: dict[str, FieldParser] | None = None,
    field_serializers: dict[str, FieldSerializer] | None = None,
):
    """
    Decorator to register a config dataclass.

    Args:
        name: Section name in config file
        field_mappings: Map dataclass field names to config file keys
        field_parsers: Custom parsers for specific fields (must accept value and ConfigParser)
        field_serializers: Custom serializers for saving fields to config file

    Example:
        @register_config("logging", field_mappings={"output": "log_output"})
        @dataclass
        class LoggerConfig:
            output: list[str] = field(default_factory=lambda: ["stdout"])
    """

    def decorator(config_class: type[T_Decorator]) -> type[T_Decorator]:
        _REGISTRY[config_class] = (
            name,
            field_mappings or {},
            field_parsers or {},
            field_serializers or {},
        )
        return config_class

    return decorator


def get_all_registered() -> list[type]:
    return list(_REGISTRY.keys())


def is_registered(config_class: type) -> bool:
    return config_class in _REGISTRY


def get_section_name(config_class: type) -> str:
    if not is_registered(config_class):
        raise ValueError(f"Config class {config_class.__name__} is not registered")
    return _REGISTRY[config_class][0]


def get_field_mappings(config_class: type) -> dict[str, str]:
    if not is_registered(config_class):
        raise ValueError(f"Config class {config_class.__name__} is not registered")
    return _REGISTRY[config_class][1]


def get_field_parsers(config_class: type) -> dict[str, FieldParser]:
    if not is_registered(config_class):
        raise ValueError(f"Config class {config_class.__name__} is not registered")
    return _REGISTRY[config_class][2]


def get_field_serializers(config_class: type) -> dict[str, FieldSerializer]:
    if not is_registered(config_class):
        raise ValueError(f"Config class {config_class.__name__} is not registered")
    return _REGISTRY[config_class][3]
