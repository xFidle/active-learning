#!/usr/bin/env bash
set -euo pipefail

for classifier in "svm" "forest"; do
  for initializer in "random" "cluster"; do
    for selector in "uncertainty" "diversity" "random"; do
      echo "Drawing plot for "$classifier"_"$initializer"_"$selector""
      uv run python -m src.plot_pr_curve "$classifier" "$initializer" "$selector" --ths 25 30 50 100
    done
  done
done
