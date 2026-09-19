import warnings
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, average_precision_score
from ..utils.logger import setup_logger

logger = setup_logger("Explainability")
warnings.filterwarnings("ignore")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


TOPO_FEATS = ['cn', 'jc', 'aai', 'pa', 'rai', 'ccn', 'cra', 'wic']
SEMANTIC_FEATS = ['sim_atc', 'sim_chemical', 'sim_mesh', 'sim_ade']
DISPLAY_MAP = {
    'sim_atc': 'ATC',
    'sim_chemical': 'Chemical',
    'sim_mesh': 'MeSH',
    'sim_ade': 'ADE'
}


def display_name(feature_name: str) -> str:
    return DISPLAY_MAP.get(feature_name, feature_name.upper())


class AblationAndSHAP:
    """Đánh giá ablation (Leave-One-Out, Group) và tính SHAP."""
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                 seed: int = 42, model_factory=None):
        self.seed = seed
        self.model_factory = model_factory or (
            lambda: RandomForestClassifier(n_estimators=100, max_depth=15, random_state=self.seed, n_jobs=-1)
        )
        self.train_df = train_df
        self.test_df = test_df
        self._prepare()

    def _prepare(self):
        drop = ['u', 'v', 'label', 'pos_split', 'community_id', 'source', 'target', 'drugbank_id']
        feats = [c for c in self.train_df.columns if c not in drop]
        self.all_feats = feats
        self.Xtr_df = self.train_df[feats].apply(pd.to_numeric, errors='coerce').fillna(0)
        self.Xte_df = self.test_df[feats].apply(pd.to_numeric, errors='coerce').fillna(0)
        self.ytr = self.train_df['label'].to_numpy(dtype=np.int32)
        self.yte = self.test_df['label'].to_numpy(dtype=np.int32)

    def _evaluate_subset(self, feature_subset: List[str]) -> Tuple[float, float]:
        Xtr = self.Xtr_df[feature_subset].to_numpy(np.float32)
        Xte = self.Xte_df[feature_subset].to_numpy(np.float32)

        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        Xte_s = sc.transform(Xte)

        m = self.model_factory()
        m.fit(Xtr_s, self.ytr)
        yp = m.predict(Xte_s)

        f1 = f1_score(self.yte, yp, zero_division=0)

        aupr = np.nan
        if hasattr(m, "predict_proba"):
            try:
                pr = m.predict_proba(Xte_s)
                pr_pos = pr[:, 1] if pr.ndim == 2 and pr.shape[1] == 2 else np.ravel(pr)
                aupr = average_precision_score(self.yte, pr_pos)
            except Exception:
                pass

        return f1, aupr

    def leave_one_out(self) -> pd.DataFrame:
        logger.info("Thực hiện Leave-One-Out Feature Ablation...")
        base_f1, base_aupr = self._evaluate_subset(self.all_feats)
        logger.info(f"Baseline (đủ {len(self.all_feats)} đặc trưng): F1={base_f1:.2f}, AUPR={base_aupr:.2f}")

        rows = []
        for f in self.all_feats:
            subset = [c for c in self.all_feats if c != f]
            f1, aupr = self._evaluate_subset(subset)
            rows.append({
                'Removed_Feature': display_name(f),
                'F1': round(float(f1), 2),
                'Delta_F1': round(float(f1 - base_f1), 2),
                'AUPR': round(float(aupr), 2),
                'Delta_AUPR': round(float(aupr - base_aupr), 2)
            })

        df_loo = pd.DataFrame(rows).sort_values('Delta_F1', ascending=True)
        return df_loo

    def group_ablation(self) -> pd.DataFrame:
        logger.info("Thực hiện Group Ablation...")
        groups = {
            'All Features (12)': self.all_feats,
            'Topology Only (8)': [f for f in TOPO_FEATS if f in self.all_feats],
            'Semantic Only (4)': [f for f in SEMANTIC_FEATS if f in self.all_feats],
            'No ADE (11)': [f for f in self.all_feats if f != 'sim_ade'],
            'No ATC (11)': [f for f in self.all_feats if f != 'sim_atc'],
            'No Chemical (11)': [f for f in self.all_feats if f != 'sim_chemical'],
            'No MeSH (11)': [f for f in self.all_feats if f != 'sim_mesh']
        }

        rows = []
        for grp_name, f_list in groups.items():
            if not f_list:
                continue
            f1, aupr = self._evaluate_subset(f_list)
            rows.append({
                'Group': grp_name,
                'Num_Features': len(f_list),
                'F1': round(float(f1), 2),
                'AUPR': round(float(aupr), 2)
            })

        return pd.DataFrame(rows).sort_values('F1', ascending=False)

    def shap_cumulative(self, sample_size: int = 2000) -> Optional[pd.DataFrame]:
        if not HAS_SHAP:
            logger.warning("Thư viện 'shap' chưa được cài đặt. Bỏ qua SHAP cumulative.")
            return None

        logger.info(f"Tính toán SHAP Cumulative trên tập Train (mẫu {sample_size})...")
        Xtr = self.Xtr_df[self.all_feats].to_numpy(np.float32)
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)

        m = self.model_factory()
        m.fit(Xtr_s, self.ytr)

        n = min(sample_size, len(Xtr_s))
        idx = np.random.RandomState(self.seed).choice(len(Xtr_s), n, replace=False)

        explainer = shap.TreeExplainer(m)
        sv = explainer.shap_values(Xtr_s[idx])

        if hasattr(sv, 'values'):
            sv = sv.values
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1] if sv.shape[2] > 1 else sv[:, :, 0]

        importance = np.abs(sv).mean(axis=0)
        ranked_feats = [f for f, _ in sorted(zip(self.all_feats, importance), key=lambda x: -x[1])]

        logger.info(f"Thứ tự đặc trưng theo SHAP: {[display_name(f) for f in ranked_feats]}")

        rows = []
        cumulative_subset = []
        for i, f in enumerate(ranked_feats, 1):
            cumulative_subset.append(f)
            f1, aupr = self._evaluate_subset(cumulative_subset)
            rows.append({
                'Step': i,
                'Added_Feature': display_name(f),
                'Total_Features': len(cumulative_subset),
                'F1': round(float(f1), 2),
                'AUPR': round(float(aupr), 2)
            })

        return pd.DataFrame(rows)
