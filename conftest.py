import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


@pytest.fixture(autouse=True)
def cleanup_config_registry():
    yield

    from src.config.base import _REGISTRY
    from src.config import LoggerConfig, ImageProcessingConfig

    actual_configs = {LoggerConfig, ImageProcessingConfig}
    test_configs = [cls for cls in _REGISTRY.keys() if cls not in actual_configs]

    for test_config in test_configs:
        del _REGISTRY[test_config]
