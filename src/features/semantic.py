import os
import ast
import numpy as np
import pandas as pd
from typing import Optional
from ..utils.logger import setup_logger

logger = setup_logger("SemanticFeatures")


def safe_parse_list(x):
    """Phân tích chuỗi hoặc danh sách an toàn chống ngoại lệ ValueError/SyntaxError."""
    if isinstance(x, list):
        return [str(i) for i in x]
    if isinstance(x, str):
        try:
            val = ast.literal_eval(x)
            if isinstance(val, list):
                return [str(i) for i in val]
        except Exception:
            pass
        clean_str = x.replace('[', '').replace(']', '').replace("'", "").replace('"', "")
        return [item.strip() for item in clean_str.split(',') if item.strip()]
    return []


class ATCFeatureGenerator:
    """
    Tính toán đặc trưng tương đồng ATC (Drug therapeutic-based similarity).
    - Input: Tệp CSV chứa cột 'atc_codes' dạng danh sách (Level 1).
    - Output: Ma trận tương đồng Cosine (Cosine Similarity).
    """
    def __init__(self, input_path: str):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Không tìm thấy tệp {input_path}")
        
        self.df = pd.read_csv(input_path)
        self.drug_ids = self.df['drugbank_id'].tolist()
        self.N = len(self.drug_ids)
        
        self.df['atc_codes'] = self.df['atc_codes'].apply(safe_parse_list)

    def compute_atc_similarity(self, output_path: Optional[str] = None) -> pd.DataFrame:
        logger.info(f"Bắt đầu tính toán tương đồng ATC cho {self.N} loại thuốc...")
        
        unique_atc = set()
        for codes in self.df['atc_codes']:
            unique_atc.update({str(c) for c in codes if str(c)})
        
        vocab = sorted(list(unique_atc))
        vocab_index = {code: i for i, code in enumerate(vocab)}
        n_features = len(vocab)
        logger.info(f"Tìm thấy {n_features} mã ATC Level 1 duy nhất trong dữ liệu.")

        if n_features == 0:
            logger.warning("Không tìm thấy mã ATC nào. Trả về ma trận đơn vị.")
            sim_matrix = np.eye(self.N)
        else:
            # 1. Ma trận tần suất nhị phân (Binary TF)
            tf_matrix = np.zeros((self.N, n_features))
            for i, codes in enumerate(self.df['atc_codes']):
                for code in codes:
                    c_str = str(code)
                    if c_str in vocab_index:
                        tf_matrix[i, vocab_index[c_str]] = 1.0

            # 2. IDF Vector
            df_counts = np.sum(tf_matrix, axis=0)
            df_counts[df_counts == 0] = 1
            idf_vector = np.log(self.N / df_counts)

            # 3. TF-IDF và Chuẩn hóa L2
            tfidf_matrix = tf_matrix * idf_vector
            norms = np.sqrt(np.sum(tfidf_matrix**2, axis=1, keepdims=True))
            norms[norms == 0] = 1.0
            tfidf_norm = tfidf_matrix / norms

            # 4. Cosine Similarity
            sim_matrix = np.dot(tfidf_norm, tfidf_norm.T)

        df_sim = pd.DataFrame(sim_matrix, index=self.drug_ids, columns=self.drug_ids)
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df_sim.to_csv(output_path)
            logger.info(f"Đã lưu ma trận tương đồng ATC tại: {output_path}")

        return df_sim


class MeSHFeatureGenerator:
    """
    Tính toán đặc trưng tương đồng danh pháp y sinh MeSH.
    - Input: Tệp CSV chứa cột 'mesh_terms' dạng danh sách.
    - Output: Ma trận tương đồng Cosine.
    """
    def __init__(self, input_path: str):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Không tìm thấy tệp {input_path}")
        
        self.df = pd.read_csv(input_path)
        self.drug_ids = self.df['drugbank_id'].tolist()
        self.N = len(self.drug_ids)
        
        self.df[target_col] = self.df[target_col].apply(safe_parse_list)
        self.target_col = target_col

    def compute_mesh_similarity(self, output_path: Optional[str] = None) -> pd.DataFrame:
        logger.info(f"Bắt đầu tính toán tương đồng MeSH cho {self.N} loại thuốc...")
        
        unique_mesh = set()
        for terms in self.df[self.target_col]:
            unique_mesh.update({str(t) for t in terms if str(t)})
        
        vocab = sorted(list(unique_mesh))
        vocab_index = {term: i for i, term in enumerate(vocab)}
        n_features = len(vocab)
        logger.info(f"Tìm thấy {n_features} thuật ngữ MeSH duy nhất.")

        if n_features == 0:
            logger.warning("Không tìm thấy thuật ngữ MeSH nào. Trả về ma trận đơn vị.")
            sim_matrix = np.eye(self.N)
        else:
            tf_matrix = np.zeros((self.N, n_features))
            for i, terms in enumerate(self.df[self.target_col]):
                for term in terms:
                    t_str = str(term)
                    if t_str in vocab_index:
                        tf_matrix[i, vocab_index[t_str]] = 1.0

            df_counts = np.sum(tf_matrix, axis=0)
            df_counts[df_counts == 0] = 1
            idf_vector = np.log(self.N / df_counts)

            tfidf_matrix = tf_matrix * idf_vector
            norms = np.sqrt(np.sum(tfidf_matrix**2, axis=1, keepdims=True))
            norms[norms == 0] = 1.0
            tfidf_norm = tfidf_matrix / norms

            sim_matrix = np.dot(tfidf_norm, tfidf_norm.T)

        df_sim = pd.DataFrame(sim_matrix, index=self.drug_ids, columns=self.drug_ids)
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df_sim.to_csv(output_path)
            logger.info(f"Đã lưu ma trận tương đồng MeSH tại: {output_path}")

        return df_sim


class ADEFeatureGenerator:
    """
    Tính toán đặc trưng tương đồng tác dụng phụ (ADE) từ SIDER.
    - Input: Tệp CSV chứa cột 'side_effect_codes' dạng danh sách mã UMLS CUI.
    - Output: Ma trận tương đồng Cosine.
    """
    def __init__(self, input_path: str, col_name: str = 'side_effect_codes'):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Không tìm thấy tệp {input_path}")
        
        self.df = pd.read_csv(input_path)
        self.drug_ids = self.df['drugbank_id'].tolist()
        self.N = len(self.drug_ids)
        self.col_name = col_name
        self.df[self.col_name] = self.df[self.col_name].apply(self._clean_codes)

    def _clean_codes(self, item):
        if isinstance(item, list):
            return [str(x) for x in item]
        if isinstance(item, str):
            try:
                val = ast.literal_eval(item)
                if isinstance(val, list):
                    return [str(x) for x in val]
            except Exception:
                pass
            clean_str = item.replace('[', '').replace(']', '').replace("'", "").replace('"', "")
            if clean_str.strip():
                return [x.strip() for x in clean_str.split(',') if x.strip()]
        return []

    def compute_ade_similarity(self, output_path: Optional[str] = None) -> pd.DataFrame:
        logger.info(f"Bắt đầu tính toán tương đồng ADE cho {self.N} loại thuốc...")
        
        unique_ade = set()
        for codes in self.df[self.col_name]:
            unique_ade.update({str(c) for c in codes if str(c)})
        
        vocab = sorted(list(unique_ade))
        vocab_index = {code: i for i, code in enumerate(vocab)}
        n_features = len(vocab)
        logger.info(f"Tìm thấy {n_features} mã tác dụng phụ ADE duy nhất.")

        if n_features == 0:
            logger.warning("Không có mã tác dụng phụ nào. Trả về ma trận đơn vị.")
            sim_matrix = np.eye(self.N)
        else:
            tf_matrix = np.zeros((self.N, n_features))
            for i, codes in enumerate(self.df[self.col_name]):
                for code in codes:
                    c_str = str(code)
                    if c_str in vocab_index:
                        tf_matrix[i, vocab_index[c_str]] = 1.0

            df_counts = np.sum(tf_matrix, axis=0)
            df_counts[df_counts == 0] = 1
            idf_vector = np.log(self.N / df_counts)

            tfidf_matrix = tf_matrix * idf_vector
            norms = np.sqrt(np.sum(tfidf_matrix**2, axis=1, keepdims=True))
            norms[norms == 0] = 1.0
            tfidf_norm = tfidf_matrix / norms

            sim_matrix = np.dot(tfidf_norm, tfidf_norm.T)

        df_sim = pd.DataFrame(sim_matrix, index=self.drug_ids, columns=self.drug_ids)
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df_sim.to_csv(output_path)
            logger.info(f"Đã lưu ma trận tương đồng ADE tại: {output_path}")

        return df_sim


class ChemicalFeatureGenerator:
    """
    Tính toán đặc trưng tương đồng cấu trúc hóa học phân tử SMILES.
    - Morgan (Circular) Fingerprint (bán kính 2, 1024-bit).
    - Độ đo tương đồng Tanimoto (Tanimoto Similarity).
    """
    def __init__(self, input_path: str, smiles_col: str = 'smiles'):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Không tìm thấy tệp {input_path}")
        
        self.df = pd.read_csv(input_path)
        if 'drugbank_id' not in self.df.columns and 'id' in self.df.columns:
            self.df.rename(columns={'id': 'drugbank_id'}, inplace=True)
            
        self.drug_ids = self.df['drugbank_id'].tolist()
        self.N = len(self.drug_ids)
        self.smiles_col = smiles_col

    def compute_chemical_similarity(self, output_path: Optional[str] = None) -> pd.DataFrame:
        logger.info(f"Bắt đầu tính toán tương đồng cấu trúc hóa học cho {self.N} loại thuốc...")

        try:
            from rdkit import Chem, DataStructs
            from rdkit.Chem import AllChem
            has_rdkit = True
        except ImportError:
            has_rdkit = False
            logger.warning(
                "Chưa cài đặt 'rdkit'. Sử dụng n-gram Jaccard similarity trên chuỗi SMILES làm cơ chế fallback."
            )

        if has_rdkit:
            mols = []
            for sm in self.df[self.smiles_col]:
                if isinstance(sm, str) and len(sm) > 0:
                    mol = Chem.MolFromSmiles(sm)
                    mols.append(mol)
                else:
                    mols.append(None)

            fps = []
            for mol in mols:
                if mol is not None:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
                    fps.append(fp)
                else:
                    fps.append(None)

            sim_matrix = np.zeros((self.N, self.N))
            for i in range(self.N):
                sim_matrix[i, i] = 1.0
                for j in range(i + 1, self.N):
                    if fps[i] is not None and fps[j] is not None:
                        sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                        sim_matrix[i, j] = sim
                        sim_matrix[j, i] = sim
                    else:
                        sim_matrix[i, j] = 0.0
                        sim_matrix[j, i] = 0.0
        else:
            # Fallback: K-mer character set Jaccard
            def sm_to_ngrams(s, n=3):
                if not isinstance(s, str) or len(s) < n:
                    return set()
                return {s[k:k+n] for k in range(len(s) - n + 1)}

            sm_sets = [sm_to_ngrams(s) for s in self.df[self.smiles_col]]
            sim_matrix = np.zeros((self.N, self.N))
            for i in range(self.N):
                sim_matrix[i, i] = 1.0
                for j in range(i + 1, self.N):
                    s1, s2 = sm_sets[i], sm_sets[j]
                    if s1 and s2:
                        jacc = len(s1 & s2) / len(s1 | s2)
                        sim_matrix[i, j] = jacc
                        sim_matrix[j, i] = jacc
                    else:
                        sim_matrix[i, j] = 0.0
                        sim_matrix[j, i] = 0.0

        df_sim = pd.DataFrame(sim_matrix, index=self.drug_ids, columns=self.drug_ids)
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df_sim.to_csv(output_path)
            logger.info(f"Đã lưu ma trận tương đồng Chemical tại: {output_path}")

        return df_sim
