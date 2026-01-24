#!/usr/bin/env bash
set -euo pipefail

initializer=""

while getopts "i:" opt; do
  case "$opt" in
  i)
    initializer="$OPTARG"
    ;;
  *)
    echo "Usage: $0 [-i cluster|random]"
    exit 1
    ;;
  esac

done

if [[ -n "$initializer" && "$initializer" != "cluster" && "$initializer" != "random" ]]; then
  echo "Initializer must be 'cluster' or 'random'."
  exit 1
fi

if [[ -n "$initializer" ]]; then
  pattern="configs/*_${initializer}_*.toml"
else
  pattern="configs/*.toml"
fi

for config in $pattern; do
  echo "Running $config"
  # uv run python -m src.main --config "$config"
done
