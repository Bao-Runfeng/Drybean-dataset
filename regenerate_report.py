"""根据已有 outputs/*.csv 重新生成图表和 report.html。"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.models import available_model_names, select_model_specs
from src.reporting import generate_static_report
from src.visualization import (
    ensure_figure_dir,
    plot_class_distribution,
    plot_cleaning_summary,
    plot_individual_class_f1,
    plot_individual_loss_curves,
    plot_individual_metric_summaries,
    plot_individual_robustness,
    plot_loss_curves,
    plot_metric_comparisons,
    plot_robustness,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="根据已有输出重新生成 Dry Bean 静态报告。"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="all 或 metrics_summary.csv 中已有的逗号分隔模型名称",
    )
    return parser.parse_args()


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    figure_dir = ensure_figure_dir(output_dir)

    metrics = read_csv_or_empty(output_dir / "metrics_summary.csv")
    if metrics.empty:
        raise FileNotFoundError(
            f"缺少 {output_dir / 'metrics_summary.csv'}，无法重新生成报告。"
        )

    if args.models == "all":
        present = set(metrics["model"])
        model_names = [name for name in available_model_names() if name in present]
        model_specs = select_model_specs(model_names)
    else:
        model_specs = select_model_specs(args.models)

    selected_names = {spec.name for spec in model_specs}
    metrics = metrics[metrics["model"].isin(selected_names)].copy()

    cleaning_stats = read_csv_or_empty(output_dir / "data_cleaning_summary.csv")
    class_distribution = read_csv_or_empty(output_dir / "class_distribution.csv")
    classification_reports = read_csv_or_empty(output_dir / "classification_reports.csv")
    robustness = read_csv_or_empty(output_dir / "robustness_results.csv")
    loss_history = read_csv_or_empty(output_dir / "loss_history.csv")

    if not classification_reports.empty:
        classification_reports = classification_reports[
            classification_reports["model"].isin(selected_names)
        ].copy()
    if not robustness.empty:
        robustness = robustness[robustness["model"].isin(selected_names)].copy()
    if not loss_history.empty:
        loss_history = loss_history[loss_history["model"].isin(selected_names)].copy()

    comparison_plot_paths = []
    if not cleaning_stats.empty:
        comparison_plot_paths.append(plot_cleaning_summary(cleaning_stats, figure_dir))
    if not class_distribution.empty:
        comparison_plot_paths.append(plot_class_distribution(class_distribution, figure_dir))
    existing_feature_corr = figure_dir / "feature_correlation.png"
    if existing_feature_corr.exists():
        comparison_plot_paths.append(existing_feature_corr)
    comparison_plot_paths.extend(plot_metric_comparisons(metrics, figure_dir))

    loss_plot = plot_loss_curves(loss_history, figure_dir)
    if loss_plot is not None:
        comparison_plot_paths.append(loss_plot)
    robust_plot = plot_robustness(robustness, figure_dir)
    if robust_plot is not None:
        comparison_plot_paths.append(robust_plot)

    individual_plot_paths = {spec.name: [] for spec in model_specs}
    for spec in model_specs:
        existing_confusion = figure_dir / f"confusion_matrix_{spec.name}.png"
        if existing_confusion.exists():
            individual_plot_paths[spec.name].append(existing_confusion)

    for plot_mapping in [
        plot_individual_metric_summaries(metrics, figure_dir),
        plot_individual_class_f1(classification_reports, figure_dir),
        plot_individual_loss_curves(loss_history, figure_dir),
        plot_individual_robustness(robustness, figure_dir),
    ]:
        for model_name, paths in plot_mapping.items():
            individual_plot_paths.setdefault(model_name, []).extend(paths)

    report_path = generate_static_report(
        output_dir=output_dir,
        cleaning_stats=cleaning_stats,
        class_distribution=class_distribution,
        metrics=metrics,
        robustness=robustness,
        classification_reports=classification_reports,
        model_specs=model_specs,
        comparison_plot_paths=comparison_plot_paths,
        individual_plot_paths=individual_plot_paths,
    )
    print(f"已重新生成报告：{report_path}")


if __name__ == "__main__":
    main()
