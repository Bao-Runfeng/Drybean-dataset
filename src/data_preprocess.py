"""数据读取、清洗和特征预处理。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .constants import (
    CANONICAL_CLASSES,
    ENGINEERED_FEATURE_COLUMNS,
    POSITIVE_FEATURES,
    RAW_FEATURE_COLUMNS,
    SPLIT_FILES,
    TARGET_COLUMN,
    ZERO_ONE_FEATURES,
)


NUMBER_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


@dataclass
class CleaningStats:
    """单个数据集划分的清洗统计。"""

    split: str
    raw_rows: int
    cleaned_rows: int
    raw_columns: int
    duplicate_rows_raw: int
    duplicate_rows_removed: int
    missing_cells_raw: int
    missing_cells_after_numeric_conversion: int
    non_numeric_feature_cells: int
    negative_area_fixed: int
    class_label_variants_normalized: int
    invalid_class_rows_dropped: int
    invalid_domain_cells_set_missing: int


def load_raw_splits(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """读取训练集、验证集和测试集 CSV 文件。"""

    data_dir = Path(data_dir)
    splits: Dict[str, pd.DataFrame] = {}
    for split, file_name in SPLIT_FILES.items():
        path = data_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"找不到 {split} 数据文件：{path}")
        # utf-8-sig 用于处理首列列名前可能存在的 BOM 字符。
        splits[split] = pd.read_csv(path, encoding="utf-8-sig")
    return splits


def normalize_class_label(value: object) -> str | float:
    """将脏标签统一为七个标准干豆类别。"""

    if pd.isna(value):
        return np.nan
    label = str(value).strip().upper()
    label = label.replace("0", "O").replace("3", "E")
    if label in CANONICAL_CLASSES:
        return label
    return np.nan


def parse_numeric_cell(value: object) -> float:
    """将一个特征单元格转换为浮点数。"""

    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    if text in {"", "?", "nan", "NaN", "None"}:
        return np.nan
    match = NUMBER_PATTERN.search(text)
    if match is None:
        return np.nan
    return float(match.group(0))


def count_non_numeric_feature_cells(df: pd.DataFrame, feature_columns: Iterable[str]) -> int:
    """统计非空但无法直接读取为数值的特征单元格数量。"""

    count = 0
    for column in feature_columns:
        numeric = pd.to_numeric(df[column], errors="coerce")
        count += int((numeric.isna() & df[column].notna()).sum())
    return count


def clean_dataframe(
    df: pd.DataFrame,
    split: str,
    remove_duplicates: bool = False,
) -> Tuple[pd.DataFrame, CleaningStats]:
    """清洗一个数据集划分，并返回清洗后的数据和统计记录。"""

    raw = df.copy()
    raw.columns = [str(col).replace("\ufeff", "").strip() for col in raw.columns]

    missing_cells_raw = int(raw.isna().sum().sum())
    duplicate_rows_raw = int(raw.duplicated().sum())
    non_numeric_cells = count_non_numeric_feature_cells(raw, RAW_FEATURE_COLUMNS)

    cleaned = raw.copy()
    class_before = cleaned[TARGET_COLUMN].copy()
    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].map(normalize_class_label)
    invalid_class_rows = int(cleaned[TARGET_COLUMN].isna().sum())

    normalized_comparison = class_before.map(normalize_class_label)
    class_variants = int(
        class_before.fillna("")
        .astype(str)
        .ne(normalized_comparison.fillna("").astype(str))
        .sum()
    )

    for column in RAW_FEATURE_COLUMNS:
        cleaned[column] = cleaned[column].map(parse_numeric_cell)

    negative_area_fixed = 0
    if "Area" in cleaned.columns:
        negative_area_fixed = int((cleaned["Area"] < 0).sum())
        cleaned.loc[cleaned["Area"] < 0, "Area"] = cleaned.loc[
            cleaned["Area"] < 0, "Area"
        ].abs()

    invalid_domain_cells = 0
    for column in POSITIVE_FEATURES:
        if column in cleaned.columns:
            mask = cleaned[column].notna() & (cleaned[column] <= 0)
            invalid_domain_cells += int(mask.sum())
            cleaned.loc[mask, column] = np.nan

    for column in ZERO_ONE_FEATURES:
        if column in cleaned.columns:
            mask = cleaned[column].notna() & ~cleaned[column].between(0, 1)
            invalid_domain_cells += int(mask.sum())
            cleaned.loc[mask, column] = np.nan

    cleaned = cleaned.dropna(subset=[TARGET_COLUMN]).copy()

    duplicate_rows_removed = 0
    if remove_duplicates:
        before = len(cleaned)
        cleaned = cleaned.drop_duplicates().copy()
        duplicate_rows_removed = before - len(cleaned)

    stats = CleaningStats(
        split=split,
        raw_rows=len(raw),
        cleaned_rows=len(cleaned),
        raw_columns=raw.shape[1],
        duplicate_rows_raw=duplicate_rows_raw,
        duplicate_rows_removed=duplicate_rows_removed,
        missing_cells_raw=missing_cells_raw,
        missing_cells_after_numeric_conversion=int(cleaned.isna().sum().sum()),
        non_numeric_feature_cells=non_numeric_cells,
        negative_area_fixed=negative_area_fixed,
        class_label_variants_normalized=class_variants,
        invalid_class_rows_dropped=invalid_class_rows,
        invalid_domain_cells_set_missing=invalid_domain_cells,
    )
    return cleaned.reset_index(drop=True), stats


def clean_splits(
    raw_splits: Dict[str, pd.DataFrame],
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """清洗全部数据集划分，并返回统计表。"""

    cleaned_splits: Dict[str, pd.DataFrame] = {}
    stats: List[CleaningStats] = []
    for split, df in raw_splits.items():
        cleaned, split_stats = clean_dataframe(
            df, split=split, remove_duplicates=(split == "train")
        )
        cleaned_splits[split] = cleaned
        stats.append(split_stats)
    return cleaned_splits, pd.DataFrame([asdict(item) for item in stats])


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """构造少量具有形状含义的组合特征。"""

    engineered = df.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        engineered["MajorMinorAxisRatio"] = (
            engineered["MajorAxisLength"] / engineered["MinorAxisLength"]
        )
        engineered["AreaConvexAreaRatio"] = engineered["Area"] / engineered["ConvexArea"]
        engineered["PerimeterSqrtAreaRatio"] = engineered["Perimeter"] / np.sqrt(
            engineered["Area"]
        )
        engineered["AxisLengthDiff"] = (
            engineered["MajorAxisLength"] - engineered["MinorAxisLength"]
        )
    engineered = engineered.replace([np.inf, -np.inf], np.nan)
    return engineered


def get_feature_columns(add_features: bool = True) -> List[str]:
    """返回模型流程使用的特征列。"""

    if add_features:
        return RAW_FEATURE_COLUMNS + ENGINEERED_FEATURE_COLUMNS
    return list(RAW_FEATURE_COLUMNS)


def split_features_and_target(
    cleaned_splits: Dict[str, pd.DataFrame],
    add_features: bool = True,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.Series], List[str]]:
    """从清洗后的数据中拆分特征和标签。"""

    feature_splits: Dict[str, pd.DataFrame] = {}
    target_splits: Dict[str, pd.Series] = {}
    feature_columns = get_feature_columns(add_features=add_features)

    for split, df in cleaned_splits.items():
        working = add_engineered_features(df) if add_features else df.copy()
        feature_splits[split] = working[feature_columns].copy()
        target_splits[split] = working[TARGET_COLUMN].copy()
    return feature_splits, target_splits, feature_columns


def encode_targets(
    target_splits: Dict[str, pd.Series],
) -> Tuple[Dict[str, np.ndarray], LabelEncoder]:
    """将类别名称编码为模型可使用的整数标签。"""

    label_encoder = LabelEncoder()
    label_encoder.fit(CANONICAL_CLASSES)
    encoded = {
        split: label_encoder.transform(series.astype(str))
        for split, series in target_splits.items()
    }
    return encoded, label_encoder


def make_feature_preprocessor() -> Pipeline:
    """创建只在训练集上拟合的数值预处理流程。"""

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def fit_transform_features(
    feature_splits: Dict[str, pd.DataFrame],
    feature_columns: List[str],
) -> Tuple[Dict[str, np.ndarray], Pipeline]:
    """在训练集上拟合缺失值填补和标准化流程，并转换全部数据集。"""

    preprocessor = make_feature_preprocessor()
    preprocessor.fit(feature_splits["train"][feature_columns])
    arrays = {
        split: preprocessor.transform(df[feature_columns])
        for split, df in feature_splits.items()
    }
    return arrays, preprocessor


def transform_with_preprocessor(
    preprocessor: Pipeline,
    df: pd.DataFrame,
    feature_columns: List[str],
) -> np.ndarray:
    """使用已拟合的预处理器转换一个特征表。"""

    return preprocessor.transform(df[feature_columns])


def build_class_distribution_table(
    cleaned_splits: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """返回标签清洗后的类别数量表。"""

    rows = []
    for split, df in cleaned_splits.items():
        counts = df[TARGET_COLUMN].value_counts().sort_index()
        for class_name, count in counts.items():
            rows.append({"split": split, "class_name": class_name, "count": int(count)})
    return pd.DataFrame(rows)
