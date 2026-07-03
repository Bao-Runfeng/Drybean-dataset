"""基于加噪训练数据的鲁棒性实验。"""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from .constants import POSITIVE_FEATURES, ZERO_ONE_FEATURES
from .data_preprocess import (
    fit_transform_features,
    transform_with_preprocessor,
)
from .evaluation import fit_model
from .models import ModelSpec


def repair_feature_domain(X: pd.DataFrame) -> pd.DataFrame:
    """修正噪声产生的不合理特征值。"""

    repaired = X.copy()
    for column in POSITIVE_FEATURES:
        if column in repaired.columns:
            repaired.loc[repaired[column] <= 0, column] = np.nan
    for column in ZERO_ONE_FEATURES:
        if column in repaired.columns:
            repaired.loc[~repaired[column].between(0, 1), column] = np.nan
    return repaired


def add_training_noise(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    noise_type: str,
    strength: float,
    random_state: int,
    class_values: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    """向训练数据加入一种噪声。"""

    rng = np.random.default_rng(random_state)
    X_noisy = X_train.copy(deep=True)
    y_noisy = y_train.copy()

    if noise_type == "gaussian":
        for column in X_noisy.columns:
            std = X_noisy[column].std(skipna=True)
            if pd.isna(std) or std == 0:
                continue
            mask = X_noisy[column].notna()
            X_noisy.loc[mask, column] = X_noisy.loc[mask, column] + rng.normal(
                loc=0.0,
                scale=strength * std,
                size=int(mask.sum()),
            )
        X_noisy = repair_feature_domain(X_noisy)

    elif noise_type == "missing":
        mask = rng.random(X_noisy.shape) < strength
        X_noisy = X_noisy.mask(mask)

    elif noise_type == "label_flip":
        n_flip = int(round(len(y_noisy) * strength))
        if n_flip > 0:
            indices = rng.choice(len(y_noisy), size=n_flip, replace=False)
            for index in indices:
                alternatives = class_values[class_values != y_noisy[index]]
                y_noisy[index] = rng.choice(alternatives)

    else:
        raise ValueError(f"不支持的噪声类型：{noise_type}")

    return X_noisy, y_noisy


def run_robustness_experiments(
    model_specs: List[ModelSpec],
    clean_feature_splits: Dict[str, pd.DataFrame],
    encoded_targets: Dict[str, np.ndarray],
    feature_columns: List[str],
    baseline_metrics: pd.DataFrame,
    class_values: np.ndarray,
    noise_types: Iterable[str],
    strengths: Iterable[float],
    random_state: int,
    epochs: int,
) -> pd.DataFrame:
    """使用加噪训练数据重新训练各模型，并评估测试集准确率。"""

    baseline_lookup = baseline_metrics.set_index("model")["test_accuracy"].to_dict()
    rows = []

    for noise_type in noise_types:
        for strength in strengths:
            X_noisy_train, y_noisy_train = add_training_noise(
                clean_feature_splits["train"],
                encoded_targets["train"],
                noise_type=noise_type,
                strength=float(strength),
                random_state=random_state + int(strength * 10000) + len(noise_type),
                class_values=class_values,
            )

            noisy_splits = {
                "train": X_noisy_train,
                "test": clean_feature_splits["test"],
            }
            arrays, preprocessor = fit_transform_features(noisy_splits, feature_columns)
            X_test = transform_with_preprocessor(
                preprocessor, clean_feature_splits["test"], feature_columns
            )

            for spec in model_specs:
                model, _ = fit_model(
                    spec,
                    arrays["train"],
                    y_noisy_train,
                    X_val=None,
                    y_val=None,
                    classes=class_values,
                    random_state=random_state,
                    epochs=epochs,
                    loss_every=max(1, epochs),
                )
                pred = model.predict(X_test)
                accuracy = float(accuracy_score(encoded_targets["test"], pred))
                baseline = float(baseline_lookup[spec.name])
                rows.append(
                    {
                        "model": spec.name,
                        "display_name": spec.display_name,
                        "noise_type": noise_type,
                        "strength": float(strength),
                        "test_accuracy": accuracy,
                        "baseline_test_accuracy": baseline,
                        "accuracy_drop": baseline - accuracy,
                    }
                )

    return pd.DataFrame(rows)
