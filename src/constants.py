"""干豆分类项目的公共常量。"""

from pathlib import Path


TARGET_COLUMN = "Class"

TRAIN_FILE = "Dry_Bean_Dataset_Dirty_train.csv"
VAL_FILE = "Dry_Bean_Dataset_Dirty_val.csv"
TEST_FILE = "Dry_Bean_Dataset_Dirty_test.csv"

SPLIT_FILES = {
    "train": TRAIN_FILE,
    "val": VAL_FILE,
    "test": TEST_FILE,
}

RAW_FEATURE_COLUMNS = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRation",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4",
]

ENGINEERED_FEATURE_COLUMNS = [
    "MajorMinorAxisRatio",
    "AreaConvexAreaRatio",
    "PerimeterSqrtAreaRatio",
    "AxisLengthDiff",
]

CANONICAL_CLASSES = [
    "BARBUNYA",
    "BOMBAY",
    "CALI",
    "DERMASON",
    "HOROZ",
    "SEKER",
    "SIRA",
]

POSITIVE_FEATURES = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRation",
    "ConvexArea",
    "EquivDiameter",
    "ShapeFactor1",
    "ShapeFactor2",
]

ZERO_ONE_FEATURES = [
    "Eccentricity",
    "Extent",
    "Solidity",
    "roundness",
    "Compactness",
    "ShapeFactor3",
    "ShapeFactor4",
    "AreaConvexAreaRatio",
]

DEFAULT_DATA_DIR = Path("DryBeanDataset")
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_MODEL_DIR = Path("models")
