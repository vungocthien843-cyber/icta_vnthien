from typing import Dict, Optional, Any
import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, log_loss, accuracy_score,
    average_precision_score, roc_auc_score
)


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                        y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Tính các chỉ số đánh giá phân loại nhị phân."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        'aupr': np.nan,
        'roc_auc': np.nan,
        'log_loss': np.nan
    }

    if y_prob is not None:
        try:
            metrics['aupr'] = float(average_precision_score(y_true, y_prob))
        except Exception:
            pass
        try:
            metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            pass
        try:
            metrics['log_loss'] = float(log_loss(y_true, y_prob, labels=[0, 1]))
        except Exception:
            pass

    return metrics
