import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class ImageProcessingConfig:
    model: Literal["resnet50", "vgg16"] = "resnet50"
    data_dir: Path = Path("data/flowers/images")
    output_dir: Path = Path("data/flowers/processed")
    force_download: bool = False

    def argparse_overrides(self, args: argparse.Namespace) -> "ImageProcessingConfig":
        overrides = {}

        if args.model is not None:
            overrides["model"] = args.model

        if args.data_dir is not None:
            overrides["data_dir"] = Path(args.data_dir)

        if args.output_dir is not None:
            overrides["output_dir"] = Path(args.output_dir)

        if args.force_download is not None:
            overrides["force_download"] = args.force_download

        for key, value in overrides.items():
            setattr(self, key, value)

        return self
