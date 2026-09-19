import os
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from ..utils.logger import setup_logger

logger = setup_logger("DatasetBuilder")


def build_final_train_test(dataset_path: str,
                           random_state: int = 42,
                           neg_test_ratio: float = 0.34,
                           drop_helper_cols: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chia train/test từ Final_Dataset_DDI.csv:
    - Cạnh dương: giữ nguyên theo pos_split.
    - Cạnh âm: chia ngẫu nhiên theo neg_test_ratio.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Không tìm thấy tệp dữ liệu tại: {dataset_path}")

    logger.info(f"Đang chia Train/Test từ: {dataset_path} (random_state={random_state}, neg_test_ratio={neg_test_ratio})")
    df = pd.read_csv(dataset_path)
    if 'pos_split' in df.columns:
        df['pos_split'] = df['pos_split'].fillna('')

    pos = df[df['label'] == 1].copy()
    neg = df[df['label'] == 0].copy()

    # Cạnh dương: Giữ nguyên cố định theo pos_split
    pos_train = pos[pos['pos_split'] == 'train']
    pos_test = pos[pos['pos_split'] == 'test']

    # Cạnh âm: Chia ngẫu nhiên
    neg_train, neg_test = train_test_split(
        neg, test_size=neg_test_ratio, random_state=random_state, shuffle=True
    )

    df_train = pd.concat([pos_train, neg_train], ignore_index=True)
    df_test = pd.concat([pos_test, neg_test], ignore_index=True)

    # Trộn ngẫu nhiên
    df_train = df_train.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    df_test = df_test.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    if drop_helper_cols:
        helper = ['u', 'v', 'pos_split']
        df_train = df_train.drop(columns=[c for c in helper if c in df_train.columns])
        df_test = df_test.drop(columns=[c for c in helper if c in df_test.columns])

    p_tr, n_tr = int((df_train['label'] == 1).sum()), int((df_train['label'] == 0).sum())
    p_te, n_te = int((df_test['label'] == 1).sum()), int((df_test['label'] == 0).sum())
    
    logger.info(f"Train set: {df_train.shape} | dương={p_tr:,}, âm={n_tr:,} (tỷ lệ âm:dương = {n_tr/max(p_tr, 1):.2f}:1)")
    logger.info(f"Test set : {df_test.shape} | dương={p_te:,}, âm={n_te:,} (tỷ lệ âm:dương = {n_te/max(p_te, 1):.2f}:1)")
    return df_train, df_test


def save_train_test(df_train: pd.DataFrame, df_test: pd.DataFrame, train_path: str, test_path: str) -> None:
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_path), exist_ok=True)
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)
    logger.info(f"Đã lưu tập train tại: {train_path}")
    logger.info(f"Đã lưu tập test tại: {test_path}")
