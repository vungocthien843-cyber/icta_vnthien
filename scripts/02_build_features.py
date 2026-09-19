import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.features.semantic import (
    ATCFeatureGenerator,
    MeSHFeatureGenerator,
    ADEFeatureGenerator,
    ChemicalFeatureGenerator
)
from src.features.topology import (
    DDIGraphBuilder,
    SimpleLouvainPipeline,
    LinkPredictionPipeline
)
from src.features.integration import DataIntegrationPipeline
from src.data.dataset_builder import build_final_train_test, save_train_test

logger = setup_logger("Step02_BuildFeatures")


def run(config_path: str = "configs/default.yaml", force: bool = False):
    cfg = load_config(config_path)

    # 1. 4 Semantic Features
    nodes_final = cfg.data.nodes_final

    if not os.path.exists(cfg.data.atc_csv) or force:
        logger.info("Tính ma trận tương đồng ATC...")
        ATCFeatureGenerator(nodes_final).compute_atc_similarity(cfg.data.atc_csv)

    if not os.path.exists(cfg.data.mesh_csv) or force:
        logger.info("Tính ma trận tương đồng MeSH...")
        MeSHFeatureGenerator(nodes_final).compute_mesh_similarity(cfg.data.mesh_csv)

    if not os.path.exists(cfg.data.ade_csv) or force:
        logger.info("Tính ma trận tương đồng ADE...")
        ADEFeatureGenerator(nodes_final).compute_ade_similarity(cfg.data.ade_csv)

    if not os.path.exists(cfg.data.chemical_csv) or force:
        logger.info("Tính ma trận tương đồng Chemical (SMILES)...")
        ChemicalFeatureGenerator(nodes_final).compute_chemical_similarity(cfg.data.chemical_csv)

    # 2. Graph & Louvain
    if not os.path.exists(cfg.data.graph_full_pkl) or force:
        logger.info("Dựng đồ thị DDI đầy đủ...")
        builder = DDIGraphBuilder(cfg.data.edges_clean, nodes_final)
        builder.build_graph()
        builder.save_graph(cfg.data.graph_full_pkl)

    if not os.path.exists(cfg.data.graph_train_pkl) or not os.path.exists(cfg.data.split_pkl) or force:
        logger.info("Chạy Louvain phát hiện cộng đồng trên G_train...")
        louvain_pipe = SimpleLouvainPipeline(
            cfg.data.graph_full_pkl,
            test_size=cfg.graph.test_size,
            random_state=cfg.graph.random_state
        )
        louvain_pipe.run_and_save(
            csv_path=cfg.data.communities_csv,
            pkl_path=cfg.data.graph_train_pkl,
            split_path=cfg.data.split_pkl
        )

    # 3. 8 Topology Features
    if not os.path.exists(cfg.data.topo_csv) or force:
        logger.info("Tính 8 đặc trưng topology...")
        topo_pipe = LinkPredictionPipeline(cfg.data.graph_train_pkl, cfg.data.split_pkl)
        topo_pipe.compute_8_features()
        topo_pipe.save_to_csv(cfg.data.topo_csv)

    # 4. Integrate 12 Features
    if not os.path.exists(cfg.data.final_dataset_csv) or force:
        logger.info("Ghép đặc trưng topology và semantic...")
        bio_dict = {
            "ATC": cfg.data.atc_csv,
            "MESH": cfg.data.mesh_csv,
            "ADE": cfg.data.ade_csv,
            "Chemical": cfg.data.chemical_csv
        }
        integrator = DataIntegrationPipeline(cfg.data.topo_csv, bio_dict)
        integrator.merge_all()
        integrator.save_dataset(cfg.data.final_dataset_csv)

    # 5. Build Final Train/Test Split
    if not os.path.exists(cfg.data.train_csv) or not os.path.exists(cfg.data.test_csv) or force:
        logger.info("Chia tập train/test...")
        df_train, df_test = build_final_train_test(
            cfg.data.final_dataset_csv,
            random_state=cfg.graph.random_state,
            neg_test_ratio=cfg.graph.test_size
        )
        save_train_test(df_train, df_test, cfg.data.train_csv, cfg.data.test_csv)

    logger.info("Hoàn tất trích xuất đặc trưng.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(args.config, args.force)
