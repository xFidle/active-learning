#!/usr/bin/env bash
set -euo pipefail

for classifier in "svm" "forest"; do
  for initializer in "random" "cluster"; do
    echo "Drawing plot for "$classifier"_"$initializer""
    uv run python -m src.plot_learning_curve "$classifier" "$initializer" 
  done
done
