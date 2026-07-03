"""模型训练、评估、损失记录和模型保存。"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hinge_loss,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle

from .models import ModelSpec


def fit_model(
    spec: ModelSpec,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None,
    y_val: np.ndarray | None,
    classes: np.ndarray,
    random_state: int,
    epochs: int = 60,
    loss_every: int = 1,
):
    """训练一个模型，并在需要时记录损失曲线。"""

    model = spec.create_estimator(random_state)
    loss_rows: List[Dict[str, float | int | str]] = []

    if spec.training_strategy in {"sgd_log_loss", "sgd_hinge"}:
        for epoch in range(1, epochs + 1):
            X_epoch, y_epoch = shuffle(
                X_train, y_train, random_state=random_state + epoch
            )
            model.partial_fit(X_epoch, y_epoch, classes=classes)

            if epoch == 1 or epoch % loss_every == 0 or epoch == epochs:
                row: Dict[str, float | int | str] = {
                    "model": spec.name,
                    "display_name": spec.display_name,
                    "epoch": epoch,
                }
                if spec.training_strategy == "sgd_log_loss":
                    train_proba = model.predict_proba(X_train)
                    row["train_loss"] = float(
                        log_loss(y_train, train_proba, labels=classes)
                    )
                    if X_val is not None and y_val is not None:
                        val_proba = model.predict_proba(X_val)
                        row["val_loss"] = float(
                            log_loss(y_val, val_proba, labels=classes)
                        )
                else:
                    train_scores = model.decision_function(X_train)
                    row["train_loss"] = float(
                        hinge_loss(y_train, train_scores, labels=classes)
                    )
                    if X_val is not None and y_val is not None:
                        val_scores = model.decision_function(X_val)
                        row["val_loss"] = float(
                            hinge_loss(y_val, val_scores, labels=classes)
                        )
                loss_rows.append(row)
    else:
        model.fit(X_train, y_train)

    return model, pd.DataFrame(loss_rows)


def measure_inference_speed(
    model,
    X_test: np.ndarray,
    repeats: int = 20,
) -> Tuple[float, float, float]:
    """统计测试集上的平均预测时间。"""

    model.predict(X_test[: min(len(X_test), 10)])
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict(X_test)
        durations.append(time.perf_counter() - start)
    durations = np.asarray(durations, dtype=float)
    mean_seconds = float(durations.mean())
    std_seconds = float(durations.std(ddof=0))
    ms_per_sample = mean_seconds * 1000.0 / len(X_test)
    return mean_seconds, std_seconds, ms_per_sample


def evaluate_model(
    spec: ModelSpec,
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    repeats: int = 20,
) -> Tuple[Dict[str, float | str], pd.DataFrame]:
    """计算准确率、F1、过拟合差异和分类报告。"""

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    mean_seconds, std_seconds, ms_per_sample = measure_inference_speed(
        model, X_test, repeats=repeats
    )

    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)

    metrics = {
        "model": spec.name,
        "display_name": spec.display_name,
        "course_source": spec.course_source,
        "train_accuracy": float(train_accuracy),
        "val_accuracy": float(accuracy_score(y_val, val_pred)),
        "test_accuracy": float(test_accuracy),
        "test_macro_f1": float(f1_score(y_test, test_pred, average="macro")),
        "test_weighted_f1": float(f1_score(y_test, test_pred, average="weighted")),
        "train_test_accuracy_gap": float(train_accuracy - test_accuracy),
        "inference_seconds_mean": mean_seconds,
        "inference_seconds_std": std_seconds,
        "inference_ms_per_sample": ms_per_sample,
    }

    report = classification_report(
        y_test,
        test_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).T.reset_index().rename(columns={"index": "label"})
    report_df.insert(0, "model", spec.name)
    return metrics, report_df


def save_model_bundle(
    model_dir: Path,
    spec: ModelSpec,
    model,
    preprocessor: Pipeline,
    label_encoder,
    feature_columns: List[str],
) -> Path:
    """保存训练好的模型及其预处理信息。"""

    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"{spec.name}_drybean.pkl"
    joblib.dump(
        {
            "model": model,
            "preprocessor": preprocessor,
            "label_encoder": label_encoder,
            "feature_columns": feature_columns,
        },
        path,
    )
    return path
