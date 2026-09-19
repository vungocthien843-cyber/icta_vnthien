from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats
from ..utils.logger import setup_logger

logger = setup_logger("StatTests")


def perform_statistical_tests(per_fold_scores: Dict[str, List[float]],
                              metric_name: str = "F1",
                              alpha: float = 0.05) -> pd.DataFrame:
    """
    Thực hiện kiểm định thống kê Paired t-test và Wilcoxon signed-rank test
    giữa mô hình có điểm trung bình cao nhất và tất cả các mô hình còn lại.
    """
    logger.info(f"Bắt đầu kiểm định thống kê trên chỉ số {metric_name} (per-fold)...")
    valid_scores = {k: v for k, v in per_fold_scores.items() if len(v) > 1}
    if not valid_scores:
        logger.warning("Không đủ dữ liệu per-fold để thực hiện kiểm định.")
        return pd.DataFrame()

    means = {k: np.mean(v) for k, v in valid_scores.items()}
    best_model = max(means, key=means.get)
    best_mean = means[best_model]
    logger.info(f"Mô hình tốt nhất: {best_model} (Mean = {best_mean:.2f})")

    a = np.array(valid_scores[best_model])
    rows = []

    for name, scores in valid_scores.items():
        if name == best_model:
            continue
        b = np.array(scores)
        if len(a) != len(b):
            continue

        diff = a - b
        delta_mean = float(np.mean(diff))

        if np.allclose(diff, 0):
            t_p = np.nan
            w_p = np.nan
            sig = "Identical"
        else:
            try:
                _, t_p = stats.ttest_rel(a, b)
            except Exception:
                t_p = np.nan

            try:
                _, w_p = stats.wilcoxon(a, b)
            except Exception:
                w_p = np.nan

            sig = "*" if (not np.isnan(t_p) and t_p < alpha) else ""

        rows.append({
            'Comparison': f"{best_model} vs {name}",
            'Delta_Mean': round(delta_mean, 2),
            't_test_p': round(t_p, 4) if not np.isnan(t_p) else np.nan,
            'wilcoxon_p': round(w_p, 4) if not np.isnan(w_p) else np.nan,
            'Significant': sig
        })

    df_stat = pd.DataFrame(rows)
    return df_stat
