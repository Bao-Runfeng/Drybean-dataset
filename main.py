"""命令行入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import DEFAULT_DATA_DIR, DEFAULT_MODEL_DIR, DEFAULT_OUTPUT_DIR
from src.data_preprocess import (
    build_class_distribution_table,
    clean_splits,
    encode_targets,
    fit_transform_features,
    load_raw_splits,
    split_features_and_target,
)
from src.evaluation import evaluate_model, fit_model, save_model_bundle
from src.models import select_model_specs
from src.reporting import generate_static_report
from src.robustness import run_robustness_experiments
from src.visualization import (
    ensure_figure_dir,
    plot_class_distribution,
    plot_cleaning_summary,
    plot_confusion_matrix,
    plot_feature_correlation,
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
        description="Dry Bean Dataset 多分类机器学习项目"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="all 或用逗号分隔的模型名称：logistic_regression,knn,linear_svm,random_forest,gaussian_nb",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="迭代模型 Logistic Regression 和 Linear SVM 的训练轮数。",
    )
    parser.add_argument(
        "--loss-every",
        type=int,
        default=1,
        help="每隔多少轮记录一次迭代模型损失。",
    )
    parser.add_argument(
        "--speed-repeats",
        type=int,
        default=20,
        help="推理速度测试时重复预测的次数。",
    )
    parser.add_argument(
        "--skip-robustness",
        action="store_true",
        help="跳过加噪训练鲁棒性实验。",
    )
    parser.add_argument(
        "--robustness-epochs",
        type=int,
        default=None,
        help="鲁棒性实验中迭代模型的训练轮数，默认使用 --epochs。",
    )
    parser.add_argument(
        "--noise-types",
        nargs="+",
        default=["gaussian", "missing", "label_flip"],
        choices=["gaussian", "missing", "label_flip"],
    )
    parser.add_argument(
        "--noise-strengths",
        nargs="+",
        type=float,
        default=[0.05, 0.10],
        help="鲁棒性实验使用的噪声强度。",
    )
    parser.add_argument(
        "--no-save-models",
        action="store_true",
        help="不保存训练后的模型文件。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    model_specs = select_model_specs(args.models)
    class_values = np.arange(7)

    print("1) 读取原始数据 ...")
    raw_splits = load_raw_splits(args.data_dir)

    print("2) 清洗数据并构建特征矩阵 ...")
    cleaned_splits, cleaning_stats = clean_splits(raw_splits)
    feature_splits, target_splits, feature_columns = split_features_and_target(
        cleaned_splits, add_features=True
    )
    encoded_targets, label_encoder = encode_targets(target_splits)
    feature_arrays, preprocessor = fit_transform_features(feature_splits, feature_columns)
    class_names = list(label_encoder.classes_)
    class_distribution = build_class_distribution_table(cleaned_splits)

    print("3) 训练并评估模型 ...")
    metrics_rows = []
    report_rows = []
    loss_history_rows = []
    figure_dir = ensure_figure_dir(args.output_dir)
    comparison_plot_paths = []
    individual_plot_paths = {spec.name: [] for spec in model_specs}

    for spec in model_specs:
        print(f"   - {spec.display_name}")
        model, loss_history = fit_model(
            spec,
            feature_arrays["train"],
            encoded_targets["train"],
            feature_arrays["val"],
            encoded_targets["val"],
            classes=class_values,
            random_state=args.random_state,
            epochs=args.epochs,
            loss_every=args.loss_every,
        )
        metrics, report_df = evaluate_model(
            spec,
            model,
            feature_arrays["train"],
            encoded_targets["train"],
            feature_arrays["val"],
            encoded_targets["val"],
            feature_arrays["test"],
            encoded_targets["test"],
            class_names=class_names,
            repeats=args.speed_repeats,
        )
        metrics_rows.append(metrics)
        report_rows.append(report_df)
        if not loss_history.empty:
            loss_history_rows.append(loss_history)

        y_test_pred = model.predict(feature_arrays["test"])
        individual_plot_paths[spec.name].append(
            plot_confusion_matrix(
                spec.name,
                spec.display_name,
                encoded_targets["test"],
                y_test_pred,
                class_names,
                figure_dir,
            )
        )

        if not args.no_save_models:
            save_model_bundle(
                args.model_dir,
                spec,
                model,
                preprocessor,
                label_encoder,
                feature_columns,
            )

    metrics_df = pd.DataFrame(metrics_rows)
    reports_df = pd.concat(report_rows, ignore_index=True)
    loss_history_df = (
        pd.concat(loss_history_rows, ignore_index=True)
        if loss_history_rows
        else pd.DataFrame()
    )

    print("4) 执行鲁棒性实验 ...")
    robustness_df = pd.DataFrame()
    if not args.skip_robustness:
        robustness_df = run_robustness_experiments(
            model_specs=model_specs,
            clean_feature_splits=feature_splits,
            encoded_targets=encoded_targets,
            feature_columns=feature_columns,
            baseline_metrics=metrics_df,
            class_values=class_values,
            noise_types=args.noise_types,
            strengths=args.noise_strengths,
            random_state=args.random_state,
            epochs=args.robustness_epochs or args.epochs,
        )
    else:
        print("   已通过 --skip-robustness 跳过鲁棒性实验")

    print("5) 保存表格和图表 ...")
    cleaning_stats.to_csv(args.output_dir / "data_cleaning_summary.csv", index=False)
    class_distribution.to_csv(args.output_dir / "class_distribution.csv", index=False)
    metrics_df.to_csv(args.output_dir / "metrics_summary.csv", index=False)
    reports_df.to_csv(args.output_dir / "classification_reports.csv", index=False)
    if not loss_history_df.empty:
        loss_history_df.to_csv(args.output_dir / "loss_history.csv", index=False)
    if not robustness_df.empty:
        robustness_df.to_csv(args.output_dir / "robustness_results.csv", index=False)

    comparison_plot_paths.extend(
        [
            plot_cleaning_summary(cleaning_stats, figure_dir),
            plot_class_distribution(class_distribution, figure_dir),
            plot_feature_correlation(feature_splits["train"], figure_dir),
        ]
    )
    comparison_plot_paths.extend(plot_metric_comparisons(metrics_df, figure_dir))
    loss_plot = plot_loss_curves(loss_history_df, figure_dir)
    if loss_plot is not None:
        comparison_plot_paths.append(loss_plot)
    robust_plot = plot_robustness(robustness_df, figure_dir)
    if robust_plot is not None:
        comparison_plot_paths.append(robust_plot)

    for plot_mapping in [
        plot_individual_metric_summaries(metrics_df, figure_dir),
        plot_individual_class_f1(reports_df, figure_dir),
        plot_individual_loss_curves(loss_history_df, figure_dir),
        plot_individual_robustness(robustness_df, figure_dir),
    ]:
        for model_name, paths in plot_mapping.items():
            individual_plot_paths.setdefault(model_name, []).extend(paths)

    report_path = generate_static_report(
        output_dir=args.output_dir,
        cleaning_stats=cleaning_stats,
        class_distribution=class_distribution,
        metrics=metrics_df,
        robustness=robustness_df,
        classification_reports=reports_df,
        model_specs=model_specs,
        comparison_plot_paths=comparison_plot_paths,
        individual_plot_paths=individual_plot_paths,
    )

    print("完成。")
    print(f"指标表：{args.output_dir / 'metrics_summary.csv'}")
    print(f"静态报告：{report_path}")


if __name__ == "__main__":
    main()
