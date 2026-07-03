"""数据分析和实验对比图表。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

plt.rcParams["font.sans-serif"] = [
    "PingFang SC",
    "Heiti SC",
    "STHeiti",
    "Songti SC",
    "Arial Unicode MS",
]
plt.rcParams["axes.unicode_minus"] = False


def ensure_figure_dir(output_dir: Path) -> Path:
    figure_dir = Path(output_dir) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    return figure_dir


def _save_current(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def plot_class_distribution(class_distribution: pd.DataFrame, figure_dir: Path) -> Path:
    pivot = class_distribution.pivot(
        index="class_name", columns="split", values="count"
    ).fillna(0)
    pivot = pivot[["train", "val", "test"]]
    pivot.plot(kind="bar", figsize=(10, 5))
    plt.title("标签清洗后的类别分布")
    plt.xlabel("干豆类别")
    plt.ylabel("样本数量")
    plt.xticks(rotation=35, ha="right")
    plt.legend(title="数据集")
    return _save_current(figure_dir / "class_distribution.png")


def plot_cleaning_summary(cleaning_stats: pd.DataFrame, figure_dir: Path) -> Path:
    columns = [
        "missing_cells_raw",
        "non_numeric_feature_cells",
        "negative_area_fixed",
        "class_label_variants_normalized",
        "duplicate_rows_removed",
    ]
    plot_df = cleaning_stats.set_index("split")[columns]
    plot_df.plot(kind="bar", figsize=(11, 5))
    plt.title("各数据集检测到的数据污染")
    plt.xlabel("数据集")
    plt.ylabel("单元格或行数量")
    plt.xticks(rotation=0)
    plt.legend(loc="upper right")
    return _save_current(figure_dir / "cleaning_summary.png")


def plot_feature_correlation(
    train_features: pd.DataFrame,
    figure_dir: Path,
) -> Path:
    corr = train_features.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.columns, fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    plt.title("特征相关性热力图")
    return _save_current(figure_dir / "feature_correlation.png")


def plot_metric_comparisons(metrics: pd.DataFrame, figure_dir: Path) -> list[Path]:
    paths: list[Path] = []
    ordered = metrics.sort_values("test_accuracy", ascending=False)

    plt.figure(figsize=(10, 5))
    x = np.arange(len(ordered))
    width = 0.26
    plt.bar(x - width, ordered["train_accuracy"], width=width, label="训练集")
    plt.bar(x, ordered["val_accuracy"], width=width, label="验证集")
    plt.bar(x + width, ordered["test_accuracy"], width=width, label="测试集")
    plt.xticks(x, ordered["display_name"], rotation=20, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("准确率")
    plt.title("准确率对比")
    plt.legend()
    paths.append(_save_current(figure_dir / "accuracy_comparison.png"))

    plt.figure(figsize=(10, 5))
    plt.bar(ordered["display_name"], ordered["inference_ms_per_sample"])
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("毫秒 / 样本")
    plt.title("推理速度对比")
    paths.append(_save_current(figure_dir / "inference_speed.png"))

    plt.figure(figsize=(10, 5))
    plt.bar(ordered["display_name"], ordered["train_test_accuracy_gap"])
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("训练集准确率 - 测试集准确率")
    plt.title("过拟合差异")
    paths.append(_save_current(figure_dir / "overfitting_gap.png"))
    return paths


def _annotate_bars(ax) -> None:
    """在柱状图上标注数值。"""

    y_min, y_max = ax.get_ylim()
    offset = (y_max - y_min) * 0.015
    for patch in ax.patches:
        height = patch.get_height()
        x = patch.get_x() + patch.get_width() / 2
        va = "bottom" if height >= 0 else "top"
        y = height + offset if height >= 0 else height - offset
        ax.text(x, y, f"{height:.4f}", ha="center", va=va, fontsize=8)


def plot_individual_metric_summaries(
    metrics: pd.DataFrame, figure_dir: Path
) -> dict[str, list[Path]]:
    """为每个算法生成独立指标图。"""

    paths: dict[str, list[Path]] = {}
    for _, row in metrics.sort_values("display_name").iterrows():
        model_name = row["model"]
        display_name = row["display_name"]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

        accuracy_labels = ["训练集", "验证集", "测试集"]
        accuracy_values = [
            row["train_accuracy"],
            row["val_accuracy"],
            row["test_accuracy"],
        ]
        axes[0].bar(accuracy_labels, accuracy_values, color=["#4267b2", "#70a37f", "#d98c45"])
        axes[0].set_ylim(0, 1.05)
        axes[0].set_ylabel("准确率")
        axes[0].set_title(f"{display_name} 准确率")
        _annotate_bars(axes[0])

        diagnostic_labels = ["宏平均 F1", "加权 F1", "训练-测试差异"]
        diagnostic_values = [
            row["test_macro_f1"],
            row["test_weighted_f1"],
            row["train_test_accuracy_gap"],
        ]
        colors = ["#587291", "#7d9d73", "#b4574c"]
        axes[1].bar(diagnostic_labels, diagnostic_values, color=colors)
        lower = min(-0.08, float(row["train_test_accuracy_gap"]) - 0.03)
        axes[1].set_ylim(lower, 1.05)
        axes[1].axhline(0, color="black", linewidth=0.8)
        axes[1].set_title(f"{display_name} F1 和过拟合分析")
        axes[1].tick_params(axis="x", rotation=15)
        _annotate_bars(axes[1])

        speed_text = (
            f"推理速度：{row['inference_ms_per_sample']:.6f} ms/样本\n"
            f"测试集平均预测：{row['inference_seconds_mean']:.6f} s"
        )
        axes[1].text(
            0.02,
            0.04,
            speed_text,
            transform=axes[1].transAxes,
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#d7dee8", "alpha": 0.9},
        )

        path = _save_current(figure_dir / f"model_metrics_{model_name}.png")
        paths.setdefault(model_name, []).append(path)
    return paths


def plot_loss_curves(loss_history: pd.DataFrame, figure_dir: Path) -> Path | None:
    if loss_history.empty:
        return None

    plt.figure(figsize=(10, 5))
    for display_name, group in loss_history.groupby("display_name"):
        group = group.sort_values("epoch")
        plt.plot(group["epoch"], group["train_loss"], label=f"{display_name} train")
        if "val_loss" in group.columns and group["val_loss"].notna().any():
            plt.plot(
                group["epoch"],
                group["val_loss"],
                linestyle="--",
                label=f"{display_name} val",
            )
    plt.xlabel("轮次")
    plt.ylabel("损失")
    plt.title("迭代模型损失曲线")
    plt.legend()
    return _save_current(figure_dir / "loss_curves.png")


def plot_individual_loss_curves(
    loss_history: pd.DataFrame, figure_dir: Path
) -> dict[str, list[Path]]:
    """为每个迭代模型生成独立损失曲线。"""

    paths: dict[str, list[Path]] = {}
    if loss_history.empty:
        return paths

    for model_name, group in loss_history.groupby("model"):
        group = group.sort_values("epoch")
        display_name = group["display_name"].iloc[0]
        plt.figure(figsize=(8, 4.8))
        plt.plot(group["epoch"], group["train_loss"], label="训练集损失")
        if "val_loss" in group.columns and group["val_loss"].notna().any():
            plt.plot(group["epoch"], group["val_loss"], linestyle="--", label="验证集损失")
        plt.xlabel("轮次")
        plt.ylabel("损失")
        plt.title(f"{display_name} 损失曲线")
        plt.legend()
        path = _save_current(figure_dir / f"loss_curve_{model_name}.png")
        paths.setdefault(model_name, []).append(path)
    return paths


def plot_robustness(robustness: pd.DataFrame, figure_dir: Path) -> Path | None:
    if robustness.empty:
        return None

    noise_types = list(robustness["noise_type"].drop_duplicates())
    fig, axes = plt.subplots(
        1,
        len(noise_types),
        figsize=(6 * len(noise_types), 5),
        squeeze=False,
        sharey=True,
    )
    for ax, noise_type in zip(axes[0], noise_types):
        group = robustness[robustness["noise_type"] == noise_type]
        for display_name, model_group in group.groupby("display_name"):
            model_group = model_group.sort_values("strength")
            ax.plot(
                model_group["strength"],
                model_group["accuracy_drop"],
                marker="o",
                label=display_name,
            )
        ax.set_title(f"噪声类型：{noise_type}")
        ax.set_xlabel("噪声强度")
        ax.set_ylabel("准确率下降")
        ax.axhline(0, color="black", linewidth=0.8)
    axes[0][-1].legend(loc="best", fontsize=8)
    return _save_current(figure_dir / "robustness_accuracy_drop.png")


def plot_individual_robustness(
    robustness: pd.DataFrame, figure_dir: Path
) -> dict[str, list[Path]]:
    """为每个算法生成独立鲁棒性图。"""

    paths: dict[str, list[Path]] = {}
    if robustness.empty:
        return paths

    for model_name, model_group in robustness.groupby("model"):
        display_name = model_group["display_name"].iloc[0]
        plt.figure(figsize=(8.5, 5))
        for noise_type, noise_group in model_group.groupby("noise_type"):
            noise_group = noise_group.sort_values("strength")
            plt.plot(
                noise_group["strength"],
                noise_group["accuracy_drop"],
                marker="o",
                label=noise_type,
            )
        plt.axhline(0, color="black", linewidth=0.8)
        plt.xlabel("噪声强度")
        plt.ylabel("准确率下降")
        plt.title(f"{display_name} 加噪训练鲁棒性")
        plt.legend(title="噪声类型")
        path = _save_current(figure_dir / f"robustness_{model_name}.png")
        paths.setdefault(model_name, []).append(path)
    return paths


def plot_individual_class_f1(
    classification_reports: pd.DataFrame, figure_dir: Path
) -> dict[str, list[Path]]:
    """为每个算法生成各类别 F1 图。"""

    paths: dict[str, list[Path]] = {}
    if classification_reports.empty:
        return paths

    excluded_labels = {"accuracy", "macro avg", "weighted avg"}
    for model_name, model_group in classification_reports.groupby("model"):
        class_rows = model_group[~model_group["label"].isin(excluded_labels)].copy()
        if class_rows.empty:
            continue
        plt.figure(figsize=(9, 4.8))
        plt.bar(class_rows["label"], class_rows["f1-score"], color="#5f7f95")
        plt.ylim(0, 1.05)
        plt.xlabel("类别")
        plt.ylabel("F1 分数")
        plt.title(f"{model_name} 各类别 F1 分数")
        plt.xticks(rotation=35, ha="right")
        _annotate_bars(plt.gca())
        path = _save_current(figure_dir / f"class_f1_{model_name}.png")
        paths.setdefault(model_name, []).append(path)
    return paths


def plot_confusion_matrix(
    model_name: str,
    display_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    figure_dir: Path,
) -> Path:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    plt.title(f"{display_name} 混淆矩阵")
    plt.xticks(rotation=35, ha="right")
    return _save_current(figure_dir / f"confusion_matrix_{model_name}.png")
