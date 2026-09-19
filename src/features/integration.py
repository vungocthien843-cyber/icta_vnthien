import os
import gc
from typing import Dict, Optional
import numpy as np
import pandas as pd
from ..utils.logger import setup_logger

logger = setup_logger("DataIntegration")


class DataIntegrationPipeline:
    """
    Ghép 4 đặc trưng tương đồng ngữ nghĩa & sinh học (ATC, MeSH, ADE, Chemical)
    vào bảng 8 đặc trưng cấu trúc topo, tạo thành bộ dữ liệu 12 đặc trưng hoàn chỉnh.
    Bảo toàn cột 'pos_split' cho bước phân chia train/test.
    """
    def __init__(self, topo_input, bio_paths_dict: Dict[str, str]):
        if isinstance(topo_input, str):
            if not os.path.exists(topo_input):
                raise FileNotFoundError(f"Không tìm thấy tệp Topo tại: {topo_input}")
            logger.info(f"Đọc dữ liệu Topo từ: {topo_input}")
            self.df_final = pd.read_csv(topo_input)
        elif isinstance(topo_input, pd.DataFrame):
            self.df_final = topo_input.copy()
        else:
            raise ValueError("topo_input phải là đường dẫn CSV hoặc pd.DataFrame.")

        if 'pos_split' in self.df_final.columns:
            self.df_final['pos_split'] = self.df_final['pos_split'].fillna('')

        self.bio_paths = bio_paths_dict

    def _lookup_matrix_values(self, df_target: pd.DataFrame, matrix: pd.DataFrame) -> np.ndarray:
        drug_to_idx = {drug: i for i, drug in enumerate(matrix.index)}
        u_indices = df_target['u'].map(drug_to_idx).fillna(-1).astype(int)
        v_indices = df_target['v'].map(drug_to_idx).fillna(-1).astype(int)
        valid_mask = (u_indices != -1) & (v_indices != -1)
        scores = np.zeros(len(df_target), dtype=np.float32)
        if valid_mask.any():
            mat_values = matrix.values
            scores[valid_mask.values] = mat_values[
                u_indices[valid_mask].values, v_indices[valid_mask].values
            ]
        return scores

    def merge_all(self) -> pd.DataFrame:
        logger.info("Bắt đầu ghép nối 4 ma trận tương đồng...")
        for name, path in self.bio_paths.items():
            if isinstance(path, pd.DataFrame):
                matrix_df = path
            elif isinstance(path, str) and os.path.exists(path):
                matrix_df = pd.read_csv(path, index_col=0)
            else:
                logger.warning(f"Không tìm thấy dữ liệu đặc trưng {name} tại {path}. Gán toàn bộ giá trị 0.0.")
                col_name = f"sim_{name.lower()}"
                self.df_final[col_name] = 0.0
                continue

            col_name = f"sim_{name.lower()}"
            self.df_final[col_name] = self._lookup_matrix_values(self.df_final, matrix_df)
            logger.info(f"Đã thêm đặc trưng '{col_name}'.")
            del matrix_df
            gc.collect()

        logger.info(f"Kích thước bộ dữ liệu sau khi ghép: {self.df_final.shape}")
        return self.df_final

    def save_dataset(self, output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.df_final.to_csv(output_path, index=False)
        logger.info(f"Đã lưu bộ dữ liệu tổng hợp 12 đặc trưng tại: {output_path}")
