import time
import warnings
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, log_loss, accuracy_score,
    average_precision_score
)

from .baselines import get_positive_proba, check_gpu_support
from ..utils.logger import setup_logger

logger = setup_logger("Ensemble")
warnings.filterwarnings("ignore")


class DDIEnsembleTrainer:
    """Huấn luyện và so sánh các mô hình Ensemble (Voting, Stacking)."""
    def __init__(self, seed: int = 42, use_gpu: bool = True):
        self.seed = seed
        self.use_gpu = use_gpu and check_gpu_support()
        self.rows: List[Dict[str, Any]] = []
        self.result_df: Optional[pd.DataFrame] = None

    def _prepare_matrix(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        drop = ['u', 'v', 'label', 'pos_split', 'community_id', 'source', 'target', 'drugbank_id']
        X = df[[c for c in df.columns if c not in drop]].apply(pd.to_numeric, errors='coerce')
        X = X.drop(columns=X.columns[X.isna().all()])
        return X.fillna(0).to_numpy(np.float32), df['label'].to_numpy(np.int32)

    def _get_base_models(self) -> List[Tuple[str, Any]]:
        if self.use_gpu:
            try:
                from cuml.ensemble import RandomForestClassifier as cuRF
                from cuml.svm import SVC as cuSVC
                from cuml.linear_model import LogisticRegression as cuLR
                rf = cuRF(n_estimators=100, max_depth=15, random_state=self.seed)
                svm = cuSVC(kernel='rbf', probability=True)
                lr = cuLR(max_iter=1000)
                logger.info("[Ensemble] Sử dụng cuML GPU cho RF, SVM, LR.")
            except Exception:
                from sklearn.ensemble import RandomForestClassifier as cuRF
                from sklearn.svm import SVC as cuSVC
                from sklearn.linear_model import LogisticRegression as cuLR
                rf = cuRF(n_estimators=100, max_depth=15, random_state=self.seed, n_jobs=-1)
                svm = cuSVC(kernel='rbf', probability=True, random_state=self.seed)
                lr = cuLR(max_iter=1000, n_jobs=-1, random_state=self.seed)
        else:
            from sklearn.ensemble import RandomForestClassifier as cuRF
            from sklearn.svm import SVC as cuSVC
            from sklearn.linear_model import LogisticRegression as cuLR
            rf = cuRF(n_estimators=100, max_depth=15, random_state=self.seed, n_jobs=-1)
            svm = cuSVC(kernel='rbf', probability=True, random_state=self.seed)
            lr = cuLR(max_iter=1000, n_jobs=-1, random_state=self.seed)

        return [
            ('rf', rf),
            ('svm', svm),
            ('nbc', GaussianNB()),
            ('lr', lr)
        ]

    def _eval_model(self, name: str, model: Any, Xte: np.ndarray, yte: np.ndarray, elapsed_time: float):
        yp = model.predict(Xte)
        if hasattr(yp, "get"):
            yp = yp.get()
        yp = np.asarray(yp).astype(int)
        pr = get_positive_proba(model, Xte)

        cm = confusion_matrix(yte, yp, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        f1 = f1_score(yte, yp, zero_division=0)
        aupr = average_precision_score(yte, pr) if pr is not None else np.nan
        prec = precision_score(yte, yp, zero_division=0)
        rec = recall_score(yte, yp, zero_division=0)
        spec = (tn / (tn + fp)) if (tn + fp) else 0.0
        acc = accuracy_score(yte, yp)

        loss = log_loss(yte, pr, labels=[0, 1]) if pr is not None else np.nan

        self.rows.append({
            'Model': name,
            'F1': round(float(f1), 2),
            'AUPR': round(float(aupr), 2) if not np.isnan(aupr) else np.nan,
            'Precision': round(float(prec), 2),
            'Recall': round(float(rec), 2),
            'Specificity': round(float(spec), 2),
            'Accuracy': round(float(acc), 2),
            'Loss': round(float(loss), 2) if not np.isnan(loss) else np.nan,
            'Time(s)': round(float(elapsed_time), 2)
        })
        au_str = f"{aupr:.2f}" if not np.isnan(aupr) else "—"
        logger.info(f"   {name:<24s} | F1: {f1:.2f} | AUPR: {au_str} | Time: {elapsed_time:.1f}s")

    def run(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
            cv: int = 5, passthrough: bool = True) -> pd.DataFrame:
        logger.info("Chuẩn bị dữ liệu cho Ensemble...")
        Xtr_raw, ytr = self._prepare_matrix(train_df)
        Xte_raw, yte = self._prepare_matrix(test_df)

        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr_raw)
        Xte = sc.transform(Xte_raw)

        self.rows = []

        # 1. Base Learners
        logger.info("Huấn luyện các mô hình cơ sở (Base Learners)...")
        bases = self._get_base_models()
        for nm, m in bases:
            t0 = time.time()
            m.fit(Xtr, ytr)
            self._eval_model(f"[Base] {nm.upper()}", m, Xte, yte, time.time() - t0)

        # 2. Hard Voting
        logger.info("Huấn luyện Hard Voting Classifier...")
        t0 = time.time()
        hv = VotingClassifier(estimators=self._get_base_models(), voting='hard', n_jobs=1)
        hv.fit(Xtr, ytr)
        self._eval_model("Hard Voting", hv, Xte, yte, time.time() - t0)

        # 3. Soft Voting
        logger.info("Huấn luyện Soft Voting Classifier...")
        t0 = time.time()
        sv = VotingClassifier(estimators=self._get_base_models(), voting='soft', n_jobs=1)
        sv.fit(Xtr, ytr)
        self._eval_model("Soft Voting", sv, Xte, yte, time.time() - t0)

        # 4. Stacking Classifier
        logger.info(f"Huấn luyện Stacking Classifier (cv={cv}, passthrough={passthrough})...")
        t0 = time.time()
        meta_learner = LogisticRegression(max_iter=1000, random_state=self.seed)
        st = StackingClassifier(
            estimators=self._get_base_models(),
            final_estimator=meta_learner,
            cv=cv,
            passthrough=passthrough,
            n_jobs=1
        )
        st.fit(Xtr, ytr)
        st_name = f"Stacking{'+Pass' if passthrough else ''}"
        self._eval_model(st_name, st, Xte, yte, time.time() - t0)

        self.result_df = pd.DataFrame(self.rows).sort_values('F1', ascending=False)
        return self.result_df
