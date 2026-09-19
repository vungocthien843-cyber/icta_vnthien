import os
import time
import json
import requests
import pandas as pd
from typing import Optional
from ..utils.logger import setup_logger

logger = setup_logger("SIDERMapper")


class SIDER_DrugMapper:
    """
    Ánh xạ mã thuốc DrugBank sang PubChem CID và liên kết với dữ liệu tác dụng phụ từ SIDER.
    Hỗ trợ cơ chế bộ nhớ đệm (caching) để tránh gọi lại API nhiều lần.
    """
    def __init__(self, drug_nodes_path: str, sider_tsv_path: str, cache_dir: Optional[str] = None):
        self.drug_nodes_path = drug_nodes_path
        self.sider_path = sider_tsv_path
        self.cache_path = os.path.join(cache_dir, "pubchem_cid_cache.json") if cache_dir else "data/processed/pubchem_cid_cache.json"
        
        self.df_drugs: Optional[pd.DataFrame] = None
        self.df_sider_raw: Optional[pd.DataFrame] = None
        self.df_nodes_with_se: Optional[pd.DataFrame] = None
        self.cid_cache = self._load_cache()

    def _load_cache(self) -> dict:
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.cid_cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Không lưu được cache: {e}")

    def _stitch_stereo_to_pubchem(self, cid_str):
        """Chuyển đổi ID STITCH sang PubChem CID."""
        if isinstance(cid_str, str) and cid_str.startswith('CID'):
            try:
                return int(cid_str[3:])
            except ValueError:
                return None
        return None

    def _get_cid_from_inchikey(self, inchikey: str) -> Optional[int]:
        """Lấy PubChem CID từ InChIKey thông qua PubChem PUG REST API (ưu tiên JSON)."""
        if inchikey in self.cid_cache:
            return self.cid_cache[inchikey]

        # 1. Thử qua JSON endpoint
        try:
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchikey}/cids/JSON"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                cid = data['IdentifierList']['CID'][0]
                self.cid_cache[inchikey] = int(cid)
                return int(cid)
        except Exception:
            pass

        return None

    def _get_cid_from_name(self, drug_id: str) -> Optional[int]:
        """Dự phòng tìm CID bằng DrugBank ID."""
        if drug_id in self.cid_cache:
            return self.cid_cache[drug_id]

        try:
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug_id}/cids/JSON"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                cid = data['IdentifierList']['CID'][0]
                self.cid_cache[drug_id] = int(cid)
                return int(cid)
        except Exception:
            pass

        return None

    def load_and_map_drugs(self) -> pd.DataFrame:
        """Giai đoạn 1: Nạp danh sách thuốc và tìm PubChem CID tương ứng."""
        logger.info(f"Nạp danh sách thuốc từ: {self.drug_nodes_path}")
        self.df_drugs = pd.read_csv(self.drug_nodes_path)

        if 'drugbank_id' not in self.df_drugs.columns and 'id' in self.df_drugs.columns:
            self.df_drugs.rename(columns={'id': 'drugbank_id'}, inplace=True)

        cids = []
        n_drugs = len(self.df_drugs)
        logger.info(f"Tìm PubChem CID cho {n_drugs} loại thuốc...")

        for index, row in self.df_drugs.iterrows():
            cid = None
            if 'inchikey' in row and pd.notna(row['inchikey']):
                cid = self._get_cid_from_inchikey(str(row['inchikey']).strip())
            if cid is None and 'drugbank_id' in row:
                cid = self._get_cid_from_name(str(row['drugbank_id']).strip())

            cids.append(cid)
            if (index + 1) % 50 == 0 or (index + 1) == n_drugs:
                logger.info(f"  -> Đã xử lý {index + 1}/{n_drugs} thuốc...")
                self._save_cache()
            time.sleep(0.1)

        self._save_cache()
        self.df_drugs['pubchem_cid'] = cids
        self.df_drugs = self.df_drugs.dropna(subset=['pubchem_cid']).copy()
        self.df_drugs['pubchem_cid'] = self.df_drugs['pubchem_cid'].astype(int)
        logger.info(f"Hoàn thành ánh xạ CID: {len(self.df_drugs)} thuốc có CID hợp lệ.")
        return self.df_drugs

    def load_sider_data(self) -> pd.DataFrame:
        """Giai đoạn 2: Nạp dữ liệu tác dụng phụ từ tệp SIDER TSV."""
        if not os.path.exists(self.sider_path):
            raise FileNotFoundError(f"Không tìm thấy tệp SIDER tại: {self.sider_path}")

        logger.info(f"Nạp dữ liệu SIDER từ: {self.sider_path}")
        columns = ['stitch_id_flat', 'stitch_id_sterio', 'umls_cui_label', 'meddra_type', 'umls_cui_meddra', 'side_effect_name']
        self.df_sider_raw = pd.read_csv(self.sider_path, sep='\t', names=columns)
        self.df_sider_raw['pubchem_cid'] = self.df_sider_raw['stitch_id_sterio'].apply(self._stitch_stereo_to_pubchem)
        self.df_sider_raw = self.df_sider_raw[['pubchem_cid', 'umls_cui_meddra', 'side_effect_name']].dropna(subset=['pubchem_cid'])
        self.df_sider_raw['pubchem_cid'] = self.df_sider_raw['pubchem_cid'].astype(int)
        return self.df_sider_raw

    def process_mapping_to_list(self) -> pd.DataFrame:
        """Giai đoạn 3: Ghép nối và gom nhóm tác dụng phụ theo từng thuốc."""
        logger.info("Ghép nối dữ liệu tác dụng phụ SIDER với bảng thuốc...")
        df_merged = self.df_drugs.merge(self.df_sider_raw, on='pubchem_cid', how='inner')

        sider_grouped = df_merged.groupby('drugbank_id')['umls_cui_meddra'].apply(lambda x: list(set(x))).reset_index()
        sider_grouped.rename(columns={'umls_cui_meddra': 'side_effect_codes'}, inplace=True)

        self.df_nodes_with_se = pd.merge(self.df_drugs, sider_grouped, on='drugbank_id', how='left')
        self.df_nodes_with_se['side_effect_codes'] = self.df_nodes_with_se['side_effect_codes'].apply(
            lambda x: x if isinstance(x, list) else []
        )

        count_se = self.df_nodes_with_se['side_effect_codes'].apply(len).gt(0).sum()
        logger.info(f"Hoàn tất: {count_se}/{len(self.df_nodes_with_se)} thuốc có dữ liệu tác dụng phụ.")
        return self.df_nodes_with_se

    def save_output(self, nodes_with_se_path: str, final_nodes_path: Optional[str] = None) -> None:
        """Lưu bảng nodes kèm side effects và bảng nodes rút gọn."""
        if self.df_nodes_with_se is not None:
            os.makedirs(os.path.dirname(nodes_with_se_path), exist_ok=True)
            self.df_nodes_with_se.to_csv(nodes_with_se_path, index=False)
            logger.info(f"Đã lưu nodes có tác dụng phụ tại: {nodes_with_se_path}")

            if final_nodes_path:
                cols_to_drop = [c for c in ['inchikey', 'pubchem_cid'] if c in self.df_nodes_with_se.columns]
                df_final = self.df_nodes_with_se.drop(columns=cols_to_drop)
                os.makedirs(os.path.dirname(final_nodes_path), exist_ok=True)
                df_final.to_csv(final_nodes_path, index=False)
                logger.info(f"Đã lưu nodes cuối cùng tại: {final_nodes_path}")
