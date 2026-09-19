import time
import gc
import warnings
from typing import Dict, Tuple, List, Optional, Any
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, log_loss, accuracy_score,
    average_precision_score
)

from ..utils.logger import setup_logger

logger = setup_logger("Baselines")
warnings.filterwarnings("ignore")


def check_gpu_support() -> bool:
    """Kiểm tra xem thư viện cuML (RAPIDS GPU) có khả dụng không."""
    try:
        from cuml.ensemble import RandomForestClassifier
        return True
    except Exception:
        return False


def get_positive_proba(model: Any, X: np.ndarray) -> Optional[np.ndarray]:
    """
    Trả về mảng 1-D xác suất dự đoán của lớp dương (Positive Class),
    tương thích an toàn với Scikit-learn, cuML, XGBoost và LightGBM.
    """
    if not hasattr(model, "predict_proba"):
        return None
    try:
        p = model.predict_proba(X)
    except Exception:
        return None

    if hasattr(p, "get"):  # Hỗ trợ cupy array từ cuML
        try:
            p = p.get()
        except Exception:
            pass

    p = np.asarray(p, dtype=np.float64)
    if p.ndim == 1:
        return p
    if p.ndim == 2:
        if p.shape[1] == 1:
            return p.ravel()
        return p[:, 1] if p.shape[1] == 2 else p[:, -1]
    return None


def get_baseline_models(seed: int = 42, use_gpu: bool = True) -> Dict[str, Any]:
    """
    Khởi tạo 8 mô hình học máy cơ sở với cơ chế GPU-first và CPU-fallback.
    """
    has_gpu = check_gpu_support() and use_gpu
    models = []

    models.append(('Decision Tree', DecisionTreeClassifier(random_state=seed, max_depth=15)))
    models.append(('Logistic Regression', LogisticRegression(max_iter=1000, n_jobs=-1, random_state=seed)))
    models.append(('Gaussian Naive Bayes', GaussianNB()))

    if has_gpu:
        try:
            from cuml.neighbors import KNeighborsClassifier as cuKNN
            from cuml.svm import SVC as cuSVC
            from cuml.ensemble import RandomForestClassifier as cuRF
            models.append(('kNN', cuKNN(n_neighbors=10)))
            models.append(('SVM', cuSVC(kernel='rbf', probability=True)))
            models.append(('Random Forest', cuRF(n_estimators=100, max_depth=15, random_state=seed)))
            logger.info("[GPU] cuML được kích hoạt cho kNN, SVM, Random Forest.")
        except Exception as e:
            logger.warning(f"Lỗi khởi tạo cuML: {e}. Fallback về CPU.")
            has_gpu = False

    if not has_gpu:
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier
        models.append(('kNN', KNeighborsClassifier(n_neighbors=10, n_jobs=-1)))
        models.append(('SVM', SVC(kernel='rbf', probability=True, random_state=seed)))
        models.append(('Random Forest', RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=seed)))

    # XGBoost
    try:
        import xgboost as xgb
        xgb_kwargs = dict(n_estimators=100, eval_metric='logloss', random_state=seed)
        if has_gpu:
            xgb_kwargs.update(tree_method='hist', device='cuda')
        models.append(('XGBoost', xgb.XGBClassifier(**xgb_kwargs)))
    except ImportError:
        logger.info("Chưa cài đặt 'xgboost'. Bỏ qua XGBoost.")

    # LightGBM
    try:
        import lightgbm as lgb
        lgb_dev = 'gpu' if has_gpu else 'cpu'
        models.append(('LightGBM', lgb.LGBMClassifier(n_estimators=100, max_depth=15, device=lgb_dev, random_state=seed, verbose=-1)))
    except ImportError:
        logger.info("Chưa cài đặt 'lightgbm'. Bỏ qua LightGBM.")

    return dict(models)


class BenchmarkPipeline:
    """
    Thực hiện đánh giá 5-Fold Cross Validation và Multi-run trên holdout test.
    """
    def __init__(self, random_state: int = 42, n_splits: int = 5, use_gpu: bool = True):
        self.seed = random_state
        self.n_splits = n_splits
        self.use_gpu = use_gpu
        self.table_data: List[Dict[str, Any]] = []
        self.f1_per_fold: Dict[str, List[float]] = {}
        self.aupr_per_fold: Dict[str, List[float]] = {}
        self.avg_cms: Dict[str, np.ndarray] = {}
        self.result_df: Optional[pd.DataFrame] = None

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        cols_drop = ['u', 'v', 'label', 'pos_split', 'community_id', 'source', 'target', 'drugbank_id']
        feats = [c for c in df.columns if c not in cols_drop]
        X = df[feats].apply(pd.to_numeric, errors='coerce')
        bad_cols = X.columns[X.isna().all()].tolist()
        if bad_cols:
            X = X.drop(columns=bad_cols)
        feat_names = list(X.columns)
        X_mat = X.fillna(0).to_numpy(dtype=np.float32)
        y_vec = df['label'].to_numpy(dtype=np.int32)
        return X_mat, y_vec, feat_names

    def run_cv(self, train_df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Bắt đầu {self.n_splits}-Fold Cross Validation...")
        X, y, feat_names = self.prepare_data(train_df)
        model_dict = get_baseline_models(seed=self.seed, use_gpu=self.use_gpu)
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

        self.table_data = []
        for name, model in model_dict.items():
            logger.info(f"--- Đang đánh giá mô hình: {name} ---")
            M = {'F1': [], 'Precision': [], 'Recall': [], 'Accuracy': [],
                 'Specificity': [], 'AUPR': [], 'Loss': [], 'Time': []}
            cms = []

            for fold, (tr, va) in enumerate(skf.split(X, y), 1):
                t0 = time.time()
                Xtr, Xva = X[tr], X[va]
                ytr, yva = y[tr], y[va]
                # Chuẩn hóa độc lập trong fold
                sc = StandardScaler()
                Xtr_s = sc.fit_transform(Xtr)
                Xva_s = sc.transform(Xva)

                try:
                    model.fit(Xtr_s, ytr)
                except Exception as e:
                    logger.warning(f"Lỗi fit {name} tại fold {fold}: {e}")
                    continue

                yp = model.predict(Xva_s)
                if hasattr(yp, "get"):
                    yp = yp.get()
                yp = np.asarray(yp).astype(int)

                proba = get_positive_proba(model, Xva_s)
                cm = confusion_matrix(yva, yp, labels=[0, 1])
                cms.append(cm)
                tn, fp, fn, tp = cm.ravel()

                M['F1'].append(f1_score(yva, yp, zero_division=0))
                M['Precision'].append(precision_score(yva, yp, zero_division=0))
                M['Recall'].append(recall_score(yva, yp, zero_division=0))
                M['Accuracy'].append(accuracy_score(yva, yp))
                M['Specificity'].append(tn / (tn + fp) if (tn + fp) else 0)

                if proba is not None:
                    try:
                        M['Loss'].append(log_loss(yva, proba, labels=[0, 1]))
                        M['AUPR'].append(average_precision_score(yva, proba))
                    except Exception:
                        M['Loss'].append(np.nan)
                        M['AUPR'].append(np.nan)
                else:
                    M['Loss'].append(np.nan)
                    M['AUPR'].append(np.nan)

                dt = time.time() - t0
                M['Time'].append(dt)

            self.f1_per_fold[name] = M['F1']
            self.aupr_per_fold[name] = [v for v in M['AUPR'] if not np.isnan(v)]
            if cms:
                self.avg_cms[name] = np.mean(cms, axis=0)

            def format_mean_std(metric_name):
                vals = [v for v in M[metric_name] if not np.isnan(v)]
                if not vals:
                    return "—"
                m = np.mean(vals)
                s = np.std(vals)
                return f"{m:.2f} ± {s:.2f}"

            loss_vals = [v for v in M['Loss'] if not np.isnan(v)]
            row = {
                'Model': name,
                'F1': format_mean_std('F1'),
                'AUPR': format_mean_std('AUPR'),
                'Precision': format_mean_std('Precision'),
                'Recall': format_mean_std('Recall'),
                'Specificity': format_mean_std('Specificity'),
                'Accuracy': format_mean_std('Accuracy'),
                'Loss': f"{np.mean(loss_vals):.2f} ± {np.std(loss_vals):.2f}" if loss_vals else "—",
                'Time(s)': f"{np.mean(M['Time']):.2f}" if M['Time'] else "—",
                '_aupr_sort': np.mean(self.aupr_per_fold[name]) if self.aupr_per_fold[name] else 0.0
            }
            self.table_data.append(row)

        df_res = pd.DataFrame(self.table_data).sort_values('_aupr_sort', ascending=False)
        df_res = df_res.drop(columns='_aupr_sort')
        self.result_df = df_res
        return self.result_df
