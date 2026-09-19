import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from ..utils.logger import setup_logger

logger = setup_logger("DrugBankParser")


class DrugBankParser:
    """
    Đọc, trích xuất và lọc dữ liệu từ tệp XML của DrugBank 5.0.
    Kết quả trả về gồm Nodes (thông tin thuốc hợp lệ) và Edges (tương tác thuốc).
    """
    def __init__(self, xml_path: str):
        self.xml_path = xml_path
        self.ns = {'db': 'http://www.drugbank.ca'}
        self.df_nodes: Optional[pd.DataFrame] = None
        self.df_edges: Optional[pd.DataFrame] = None

    def _get_chemical_info(self, drug_element):
        """Trích xuất ID, Tên, SMILES và InChIKey."""
        try:
            drug_id = drug_element.find("db:drugbank-id[@primary='true']", self.ns).text
            name = drug_element.find("db:name", self.ns).text
        except AttributeError:
            return None

        smiles = None
        inchikey = None

        for prop in drug_element.findall("db:calculated-properties/db:property", self.ns):
            kind_elem = prop.find("db:kind", self.ns)
            val_elem = prop.find("db:value", self.ns)
            if kind_elem is not None and val_elem is not None:
                kind = kind_elem.text
                val = val_elem.text
                if kind == "SMILES":
                    smiles = val
                elif kind == "InChIKey":
                    inchikey = val

        if not smiles or not inchikey:
            return None

        return {
            'drugbank_id': drug_id,
            'name': name,
            'smiles': smiles,
            'inchikey': inchikey
        }

    def _get_atc_codes(self, drug_element):
        """Lấy danh sách mã ATC Level 1 (ký tự đầu tiên đại diện nhóm giải phẫu chính)."""
        atc_list = []
        for atc in drug_element.findall("db:atc-codes/db:atc-code", self.ns):
            code = atc.get('code')
            if code:
                atc_list.append(code[0])
        return list(set(atc_list))

    def _get_mesh_terms(self, drug_element):
        """Lấy danh sách MeSH ID."""
        mesh_list = []
        for cat in drug_element.findall("db:categories/db:category", self.ns):
            mesh_id_elem = cat.find("db:mesh-id", self.ns)
            if mesh_id_elem is not None and mesh_id_elem.text:
                mesh_list.append(mesh_id_elem.text)
        return mesh_list

    def _get_interactions(self, drug_element, current_drug_id):
        """Trích xuất các cạnh tương tác."""
        edges = []
        for interaction in drug_element.findall("db:drug-interactions/db:drug-interaction", self.ns):
            target_id_elem = interaction.find("db:drugbank-id", self.ns)
            if target_id_elem is not None and target_id_elem.text:
                edges.append({
                    'source': current_drug_id,
                    'target': target_id_elem.text
                })
        return edges

    def parse(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Thực thi phân tích cú pháp XML."""
        if not os.path.exists(self.xml_path):
            raise FileNotFoundError(f"Không tìm thấy tệp DrugBank XML tại: {self.xml_path}")

        logger.info(f"Bắt đầu xử lý DrugBank XML: {self.xml_path}")
        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        nodes = []
        all_edges = []

        for drug in root.findall('db:drug', self.ns):
            chem_info = self._get_chemical_info(drug)
            if chem_info is None:
                continue

            atc = self._get_atc_codes(drug)
            if not atc:
                continue

            mesh = self._get_mesh_terms(drug)
            if not mesh:
                continue

            node_data = chem_info.copy()
            node_data['atc_codes'] = atc
            node_data['mesh_terms'] = mesh
            nodes.append(node_data)

            drug_interactions = self._get_interactions(drug, chem_info['drugbank_id'])
            all_edges.extend(drug_interactions)

        self.df_nodes = pd.DataFrame(nodes)

        # Lọc cạnh: Chỉ giữ tương tác giữa các thuốc hợp lệ
        valid_ids = set(self.df_nodes['drugbank_id'])
        self.df_edges = pd.DataFrame(all_edges)
        if not self.df_edges.empty:
            self.df_edges = self.df_edges[self.df_edges['target'].isin(valid_ids)]
            self.df_edges = self.df_edges.drop_duplicates()

        logger.info(f"Hoàn thành trích xuất: {len(self.df_nodes)} Nodes, {len(self.df_edges)} Edges.")
        return self.df_nodes, self.df_edges

    def save_to_csv(self, nodes_path: str, edges_path: str) -> None:
        if self.df_nodes is not None and self.df_edges is not None:
            os.makedirs(os.path.dirname(nodes_path), exist_ok=True)
            os.makedirs(os.path.dirname(edges_path), exist_ok=True)
            self.df_nodes.to_csv(nodes_path, index=False)
            self.df_edges.to_csv(edges_path, index=False)
            logger.info(f"Đã lưu: {nodes_path} và {edges_path}")


class DDI_NetworkFilter:
    """
    Làm sạch mạng lưới DDI:
    1. Chuẩn hóa cạnh vô hướng (u < v).
    2. Loại bỏ tương tác tự thân (self-loop).
    3. Loại bỏ các nút cô lập (không tham gia tương tác).
    """
    def __init__(self, nodes_csv_path: str, edges_csv_path: str):
        self.nodes_path = nodes_csv_path
        self.edges_path = edges_csv_path
        self.df_nodes: Optional[pd.DataFrame] = None
        self.df_edges: Optional[pd.DataFrame] = None

    def load_data(self) -> None:
        self.df_nodes = pd.read_csv(self.nodes_path)
        self.df_edges = pd.read_csv(self.edges_path)
        logger.info(f"Input: {len(self.df_nodes)} nodes, {len(self.df_edges)} edges.")

    def filter_and_standardize(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.df_nodes is None or self.df_edges is None:
            self.load_data()

        valid_ids = set(self.df_nodes['drugbank_id'])
        mask = (self.df_edges['source'].isin(valid_ids)) & (self.df_edges['target'].isin(valid_ids))
        df_clean_edges = self.df_edges[mask].copy()

        # Chuẩn hóa vô hướng (sắp xếp source < target theo thứ tự chuỗi)
        edge_vals = df_clean_edges[['source', 'target']].values
        df_clean_edges[['source', 'target']] = np.sort(edge_vals, axis=1)
        df_clean_edges = df_clean_edges.drop_duplicates(subset=['source', 'target'])

        # Loại bỏ self-loop
        df_clean_edges = df_clean_edges[df_clean_edges['source'] != df_clean_edges['target']]
        self.df_edges = df_clean_edges

        # Lọc nút: Chỉ giữ lại các thuốc có trong danh sách cạnh sạch
        active_ids = set(self.df_edges['source']).union(set(self.df_edges['target']))
        self.df_nodes = self.df_nodes[self.df_nodes['drugbank_id'].isin(active_ids)].copy()

        logger.info(f"Kết quả sau lọc: {len(self.df_nodes)} nodes, {len(self.df_edges)} edges.")
        return self.df_nodes, self.df_edges

    def save_to_csv(self, nodes_out: str, edges_out: str) -> None:
        if self.df_nodes is not None and self.df_edges is not None:
            os.makedirs(os.path.dirname(nodes_out), exist_ok=True)
            os.makedirs(os.path.dirname(edges_out), exist_ok=True)
            self.df_nodes.to_csv(nodes_out, index=False)
            self.df_edges.to_csv(edges_out, index=False)
            logger.info(f"Đã lưu kết quả lọc vào: {nodes_out} và {edges_out}")
