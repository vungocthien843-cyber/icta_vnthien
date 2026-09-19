import os
import pickle
import itertools
from typing import Optional, Dict, Tuple, Set
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
from ..utils.logger import setup_logger

logger = setup_logger("TopologyFeatures")


class DDIGraphBuilder:
    """
    Xây dựng và phân tích đồ thị tương tác thuốc (DDI Network).
    """
    def __init__(self, edges_file: str, nodes_file: str):
        self.edges_file = edges_file
        self.nodes_file = nodes_file
        self.G: Optional[nx.Graph] = None
        self.df_edges: Optional[pd.DataFrame] = None
        self.df_nodes: Optional[pd.DataFrame] = None

    def build_graph(self) -> nx.Graph:
        logger.info("Bắt đầu xây dựng đồ thị DDI...")
        if not os.path.exists(self.edges_file):
            raise FileNotFoundError(f"Không tìm thấy tệp edges: {self.edges_file}")
        if not os.path.exists(self.nodes_file):
            raise FileNotFoundError(f"Không tìm thấy tệp nodes: {self.nodes_file}")

        self.df_edges = pd.read_csv(self.edges_file)
        self.df_nodes = pd.read_csv(self.nodes_file)
        logger.info(f"Đọc {len(self.df_edges)} edges và {len(self.df_nodes)} nodes.")

        self.G = nx.from_pandas_edgelist(
            self.df_edges, source='source', target='target', create_using=nx.Graph()
        )

        if 'drugbank_id' in self.df_nodes.columns:
            node_attr_dict = self.df_nodes.set_index('drugbank_id').to_dict('index')
            nx.set_node_attributes(self.G, node_attr_dict)
            logger.info("Đã gắn thuộc tính node thành công.")
        else:
            logger.warning("Không tìm thấy cột 'drugbank_id' để gắn thuộc tính node.")

        return self.G

    def analyze_graph(self) -> Dict[str, float]:
        if self.G is None:
            raise ValueError("Đồ thị chưa được khởi tạo. Hãy gọi build_graph() trước.")

        n_nodes = self.G.number_of_nodes()
        n_edges = self.G.number_of_edges()
        density = nx.density(self.G)
        isolates = list(nx.isolates(self.G))

        logger.info(f"Số lượng thuốc (Nodes): {n_nodes}")
        logger.info(f"Số lượng tương tác (Edges): {n_edges}")
        logger.info(f"Mật độ đồ thị (Density): {density:.6f}")
        logger.info(f"Số lượng thuốc cô lập: {len(isolates)}")

        return {
            "num_nodes": n_nodes,
            "num_edges": n_edges,
            "density": density,
            "num_isolates": len(isolates)
        }

    def save_graph(self, output_path: str) -> None:
        if self.G is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'wb') as f:
                pickle.dump(self.G, f)
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            logger.info(f"Đã lưu đồ thị đầy đủ vào: {output_path} ({size_mb:.2f} MB)")


class SimpleLouvainPipeline:
    """
    Phân cụm cộng đồng Louvain trên G_train:
    - Tách cạnh dương thành train/test theo test_size.
    - Dựng G_train chỉ chứa các cạnh dương thuộc tập train.
    - Chạy Louvain trên G_train.
    """
    def __init__(self, graph_input, test_size: float = 0.34, random_state: int = 42):
        if isinstance(graph_input, str) and os.path.exists(graph_input):
            with open(graph_input, 'rb') as f:
                self.G_full = pickle.load(f)
        elif isinstance(graph_input, nx.Graph):
            self.G_full = graph_input
        else:
            raise ValueError("graph_input phải là đường dẫn file pkl hoặc đối tượng nx.Graph hợp lệ.")

        self.test_size = test_size
        self.random_state = random_state
        self.G_train: Optional[nx.Graph] = None
        self.partition: Optional[Dict[str, int]] = None
        self.train_pos_edges: Optional[list] = None
        self.test_pos_edges: Optional[list] = None

    def _split_and_build_train(self) -> None:
        all_pos = [tuple(sorted(e)) for e in self.G_full.edges()]
        train_e, test_e = train_test_split(
            all_pos, test_size=self.test_size,
            random_state=self.random_state, shuffle=True
        )
        self.train_pos_edges = [frozenset(e) for e in train_e]
        self.test_pos_edges = [frozenset(e) for e in test_e]

        self.G_train = nx.Graph()
        self.G_train.add_nodes_from(self.G_full.nodes(data=True))
        self.G_train.add_edges_from(train_e)

        logger.info(f"Cạnh dương Train: {len(train_e):,} | Test: {len(test_e):,}")
        logger.info(f"Đồ thị G_train: {self.G_train.number_of_nodes()} nodes, {self.G_train.number_of_edges()} edges.")

    def run_and_save(self,
                     csv_path: Optional[str] = None,
                     pkl_path: Optional[str] = None,
                     split_path: Optional[str] = None) -> pd.DataFrame:
        self._split_and_build_train()

        # Thuật toán Louvain: ưu tiên python-louvain, tự động fallback networkx built-in
        try:
            import community.community_louvain as community_louvain
            self.partition = community_louvain.best_partition(self.G_train, random_state=self.random_state)
            q_score = community_louvain.modularity(self.partition, self.G_train)
        except ImportError:
            logger.info("Không có package 'python-louvain', sử dụng 'nx.community.louvain_communities'...")
            comms = nx.community.louvain_communities(self.G_train, seed=self.random_state)
            self.partition = {node: i for i, c in enumerate(comms) for node in c}
            q_score = nx.community.modularity(self.G_train, comms)

        num_comm = len(set(self.partition.values()))
        logger.info(f"Kết quả phân cụm: {num_comm} cộng đồng. Modularity (Q) = {q_score:.4f}")

        # Gắn nhãn community vào node của G_train
        nx.set_node_attributes(self.G_train, self.partition, 'community')

        df_comm = pd.DataFrame.from_dict(self.partition, orient='index', columns=['community_id'])
        df_comm.index.name = 'drugbank_id'

        if csv_path:
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            df_comm.to_csv(csv_path)
            logger.info(f"Đã lưu bảng phân cụm tại: {csv_path}")

        if pkl_path:
            os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
            with open(pkl_path, 'wb') as f:
                pickle.dump(self.G_train, f)
            logger.info(f"Đã lưu G_train có nhãn community tại: {pkl_path}")

        if split_path:
            os.makedirs(os.path.dirname(split_path), exist_ok=True)
            with open(split_path, 'wb') as f:
                pickle.dump({'train_pos': self.train_pos_edges,
                             'test_pos': self.test_pos_edges}, f)
            logger.info(f"Đã lưu phân tách cạnh dương tại: {split_path}")

        return df_comm


class LinkPredictionPipeline:
    """
    Tính 8 đặc trưng topology trên G_train cho các cặp thuốc:
    - 5 đặc trưng lân cận: CN, JC, AAI, RAI, PA
    - 3 đặc trưng cộng đồng: CCN, CRA, WIC
    """
    def __init__(self, graph_input, split_input):
        if isinstance(graph_input, str) and os.path.exists(graph_input):
            with open(graph_input, 'rb') as f:
                self.G = pickle.load(f)
        elif isinstance(graph_input, nx.Graph):
            self.G = graph_input
        else:
            raise ValueError("graph_input không hợp lệ.")

        if isinstance(split_input, str) and os.path.exists(split_input):
            with open(split_input, 'rb') as f:
                sp = pickle.load(f)
        elif isinstance(split_input, dict):
            sp = split_input
        else:
            raise ValueError("split_input không hợp lệ.")

        self.train_pos: Set[frozenset] = set(sp['train_pos'])
        self.test_pos: Set[frozenset] = set(sp['test_pos'])
        self.df_features: Optional[pd.DataFrame] = None

        first_node = list(self.G.nodes())[0]
        if 'community' not in self.G.nodes[first_node]:
            raise ValueError("[LỖI] Đồ thị chưa được gán nhãn thuộc tính 'community'!")

    def compute_8_features(self) -> pd.DataFrame:
        nodes = list(self.G.nodes())
        all_pairs = list(itertools.combinations(nodes, 2))
        logger.info(f"Tính 8 đặc trưng topo cho {len(all_pairs):,} cặp thuốc...")

        self.df_features = pd.DataFrame(all_pairs, columns=['u', 'v'])

        # 1. Nhóm lân cận (Neighborhood)
        logger.info("Đang tính nhóm lân cận: CN, RAI, JC, AAI, PA...")
        self.df_features['cn'] = [len(list(nx.common_neighbors(self.G, u, v))) for u, v in all_pairs]
        self.df_features['rai'] = [p[2] for p in nx.resource_allocation_index(self.G, all_pairs)]
        self.df_features['jc'] = [p[2] for p in nx.jaccard_coefficient(self.G, all_pairs)]
        self.df_features['aai'] = [p[2] for p in nx.adamic_adar_index(self.G, all_pairs)]
        self.df_features['pa'] = [p[2] for p in nx.preferential_attachment(self.G, all_pairs)]

        # 2. Nhóm cộng đồng (Community-aware)
        logger.info("Đang tính nhóm cộng đồng: CCN, CRA, WIC...")
        self.df_features['ccn'] = [p[2] for p in nx.cn_soundarajan_hopcroft(self.G, all_pairs, community='community')]
        self.df_features['cra'] = [p[2] for p in nx.ra_index_soundarajan_hopcroft(self.G, all_pairs, community='community')]
        self.df_features['wic'] = [p[2] for p in nx.within_inter_cluster(self.G, all_pairs, community='community')]

        # 3. Gán nhãn và cờ split
        labels = []
        splits = []
        for u, v in all_pairs:
            fs = frozenset((u, v))
            if fs in self.train_pos:
                labels.append(1)
                splits.append('train')
            elif fs in self.test_pos:
                labels.append(1)
                splits.append('test')
            else:
                labels.append(0)
                splits.append('')

        self.df_features['label'] = labels
        self.df_features['pos_split'] = splits

        p_cnt = sum(labels)
        n_cnt = len(labels) - p_cnt
        logger.info(f"Hoàn thành: Dương={p_cnt:,}, Âm={n_cnt:,}")
        return self.df_features

    def save_to_csv(self, output_path: str) -> None:
        if self.df_features is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            self.df_features.to_csv(output_path, index=False)
            logger.info(f"Đã lưu bảng 8 đặc trưng topo tại: {output_path}")
