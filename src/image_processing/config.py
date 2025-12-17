from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class ImageProcessingConfig:
    model: Literal["resnet50", "vgg16"] = "resnet50"
    data_dir: Path = Path("data/flowers/images")
    output_dir: Path = Path("data/flowers/processed")
    force_download: bool = False
