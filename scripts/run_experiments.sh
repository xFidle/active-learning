#!/usr/bin/env bash
set -euo pipefail

for config in configs/forest*.toml; do
  echo "Running $config"
  uv run python -m src.main --config "$config"
done
