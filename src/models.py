"""模型定义。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


@dataclass(frozen=True)
class ModelSpec:
    """单个分类器的配置。"""

    name: str
    display_name: str
    course_source: str
    description: str
    training_strategy: str
    create_estimator: Callable[[int], BaseEstimator]


def _logistic_regression(random_state: int) -> BaseEstimator:
    return SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        learning_rate="optimal",
        average=True,
        random_state=random_state,
    )


def _linear_svm(random_state: int) -> BaseEstimator:
    return SGDClassifier(
        loss="hinge",
        penalty="l2",
        alpha=1e-4,
        learning_rate="optimal",
        average=True,
        random_state=random_state,
    )


def _knn(_: int) -> BaseEstimator:
    return KNeighborsClassifier(n_neighbors=7, weights="distance", metric="minkowski")


def _random_forest(random_state: int) -> BaseEstimator:
    return RandomForestClassifier(
        n_estimators=150,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )


def _gaussian_nb(_: int) -> BaseEstimator:
    return GaussianNB()


MODEL_SPECS: Dict[str, ModelSpec] = {
    "logistic_regression": ModelSpec(
        name="logistic_regression",
        display_name="Logistic Regression",
        course_source="课堂已学",
        description="线性多分类模型；使用 log loss 训练，可输出概率并绘制 loss 曲线。",
        training_strategy="sgd_log_loss",
        create_estimator=_logistic_regression,
    ),
    "knn": ModelSpec(
        name="knn",
        display_name="KNN",
        course_source="课堂已学",
        description="基于距离投票的非参数算法；天然支持多分类，没有迭代 loss 曲线。",
        training_strategy="standard",
        create_estimator=_knn,
    ),
    "linear_svm": ModelSpec(
        name="linear_svm",
        display_name="Linear SVM",
        course_source="课堂已学",
        description="使用 hinge loss 的线性支持向量机；可绘制训练过程中的 hinge loss。",
        training_strategy="sgd_hinge",
        create_estimator=_linear_svm,
    ),
    "random_forest": ModelSpec(
        name="random_forest",
        display_name="Random Forest",
        course_source="课堂未讲",
        description="多棵决策树投票的集成算法，通常对异常值和特征尺度较稳健。",
        training_strategy="standard",
        create_estimator=_random_forest,
    ),
    "gaussian_nb": ModelSpec(
        name="gaussian_nb",
        display_name="Gaussian Naive Bayes",
        course_source="课堂未讲",
        description="基于条件独立假设的概率分类器，训练和推理速度很快。",
        training_strategy="standard",
        create_estimator=_gaussian_nb,
    ),
}


def available_model_names() -> List[str]:
    """返回默认展示顺序下的模型名称。"""

    return list(MODEL_SPECS.keys())


def select_model_specs(model_names: str | Iterable[str]) -> List[ModelSpec]:
    """将模型选择参数转换为模型配置对象。"""

    if isinstance(model_names, str):
        if model_names.lower() == "all":
            names = available_model_names()
        else:
            names = [name.strip() for name in model_names.split(",") if name.strip()]
    else:
        names = list(model_names)

    unknown = [name for name in names if name not in MODEL_SPECS]
    if unknown:
        raise ValueError(
            "未知模型："
            + ", ".join(unknown)
            + "。可选模型："
            + ", ".join(available_model_names())
        )
    return [MODEL_SPECS[name] for name in names]
