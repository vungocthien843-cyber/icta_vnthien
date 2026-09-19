import os
import pickle
import pytest
import networkx as nx
from src.utils.config import load_config


def test_no_data_leakage_in_training_graph():
    """Kiểm tra đồ thị huấn luyện G_train không chứa cạnh thuộc tập test."""
    cfg = load_config("configs/default.yaml")

    split_file = cfg.data.split_pkl
    train_graph_file = cfg.data.graph_train_pkl

    assert os.path.exists(split_file), f"Không tìm thấy file: {split_file}"
    assert os.path.exists(train_graph_file), f"Không tìm thấy file: {train_graph_file}"

    with open(split_file, 'rb') as f:
        sp = pickle.load(f)
    
    train_pos = set(sp['train_pos'])
    test_pos = set(sp['test_pos'])

    # Cạnh dương train và test phải rời nhau
    overlap = train_pos.intersection(test_pos)
    assert len(overlap) == 0, f"Có {len(overlap)} cạnh trùng giữa train và test."

    # G_train không chứa cạnh test
    with open(train_graph_file, 'rb') as f:
        G_train = pickle.load(f)

    for edge in test_pos:
        u, v = tuple(edge)
        assert not G_train.has_edge(u, v), f"Cạnh test ({u}, {v}) nằm trong G_train."

    # Các node trong G_train đều có nhãn community
    for node in G_train.nodes():
        assert 'community' in G_train.nodes[node], f"Node {node} thiếu thuộc tính community."
