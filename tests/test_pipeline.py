import os
import pandas as pd
import numpy as np
import pytest
from src.utils.config import load_config


def test_dataset_integrity():
    """Kiểm tra dữ liệu train/test đủ 12 đặc trưng, không có NaN/Inf, nhãn hợp lệ."""
    cfg = load_config("configs/default.yaml")

    train_path = cfg.data.train_csv
    test_path = cfg.data.test_csv

    assert os.path.exists(train_path), f"Không tìm thấy file train: {train_path}"
    assert os.path.exists(test_path), f"Không tìm thấy file test: {test_path}"

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    expected_features = set(cfg.features.semantic + cfg.features.topology)

    for name, df in [("Train", df_train), ("Test", df_test)]:
        assert 'label' in df.columns, f"Tập {name} thiếu cột 'label'"
        
        for forbidden_col in ['u', 'v', 'pos_split']:
            assert forbidden_col not in df.columns, f"Tập {name} còn chứa cột '{forbidden_col}'"

        feature_cols = set(df.columns) - {'label'}
        assert expected_features.issubset(feature_cols), f"Tập {name} thiếu đặc trưng: {expected_features - feature_cols}"

        assert not df[list(expected_features)].isna().any().any(), f"Tập {name} có giá trị NaN"
        assert not np.isinf(df[list(expected_features)].to_numpy()).any(), f"Tập {name} có giá trị Inf"

        unique_labels = set(df['label'].unique())
        assert unique_labels.issubset({0, 1}), f"Tập {name} có nhãn không hợp lệ: {unique_labels}"
