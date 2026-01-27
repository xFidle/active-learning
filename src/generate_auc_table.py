import argparse
from pathlib import Path
from typing import cast

import pandas as pd

TARGET_RATIOS = [0.25, 0.30, 0.50, 1.00]

MODEL_LABELS = {"forest": "RF", "svm": "SVM"}

INIT_LABELS = {"random": "rand", "cluster": "clus"}

SELECTOR_LABELS = {"random": "rand", "diversity": "div", "uncertainty": "unc"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PR AUC summary table.")
    parser.add_argument("--results-dir", default="results", help="Path to results directory")
    parser.add_argument("--output-csv", default="results/auc-table.csv", help="Output CSV path")
    parser.add_argument(
        "--output-tex", default="results/auc-table.tex", help="Output LaTeX table path"
    )
    return parser.parse_args()


def label_from_dir(name: str) -> str:
    parts = name.split("_")
    if len(parts) != 3:
        return name
    model, initializer, selector = parts
    model_label = MODEL_LABELS.get(model, model)
    init_label = INIT_LABELS.get(initializer, initializer)
    selector_label = SELECTOR_LABELS.get(selector, selector)
    return f"{model_label}-{init_label}-{selector_label}"


def load_auc_means(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[["labeled_ratio", "mean"]]


def pick_nearest(df: pd.DataFrame, target: float) -> float:
    value = df.loc[(df["labeled_ratio"] - target).abs().idxmin(), "mean"]
    return cast(float, value)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_csv = Path(args.output_csv)
    output_tex = Path(args.output_tex)

    rows: list[list[float | str]] = []
    for auc_dir in sorted(results_dir.iterdir()):
        if not auc_dir.is_dir():
            continue
        if auc_dir.name in {"compare", "plots"}:
            continue
        auc_file = auc_dir / "auc-results.csv"
        if not auc_file.exists():
            continue

        df = load_auc_means(auc_file)
        values = [round(pick_nearest(df, target), 3) for target in TARGET_RATIOS]
        row: list[float | str] = [label_from_dir(auc_dir.name), *values]
        rows.append(row)

    table_df = pd.DataFrame(rows, columns=["wariant", "25\\%", "30\\%", "50\\%", "100\\%"])
    table_df = table_df.sort_values("wariant")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    table_df.to_csv(output_csv, index=False)

    output_tex.parent.mkdir(parents=True, exist_ok=True)
    latex_table = table_df.to_latex(
        index=False, float_format="%.3f", column_format=r"@{\extracolsep{\fill}}lrrrr@{}"
    )

    latex_table = latex_table.replace(r"\begin{tabular}", r"\begin{tabular*}{\linewidth}").replace(
        r"\end{tabular}", r"\end{tabular*}"
    )

    latex_table = (
        r"\begin{table}[h]"
        "\n"
        r"\centering"
        "\n" + latex_table + r"\end{table}"
        "\n"
    )

    output_tex.write_text(latex_table)

    print(latex_table)
    print(f"Saved CSV to {output_csv}")
    print(f"Saved LaTeX to {output_tex}")


if __name__ == "__main__":
    main()
