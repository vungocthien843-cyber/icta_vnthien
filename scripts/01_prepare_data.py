import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.data.parser import DrugBankParser, DDI_NetworkFilter
from src.data.sider_mapper import SIDER_DrugMapper

logger = setup_logger("Step01_PrepareData")


def run(config_path: str = "configs/default.yaml", force: bool = False):
    cfg = load_config(config_path)
    
    # Kiểm tra nếu đã có kết quả cuối cùng của Bước 1 (nodes_final)
    nodes_final = cfg.data.nodes_final
    if os.path.exists(nodes_final) and not force:
        logger.info(f"Đã có dữ liệu nodes_final tại {nodes_final}. Bỏ qua tiền xử lý thô.")
        return

    # 1. Parse DrugBank XML
    nodes_step1 = cfg.data.nodes_step1
    edges_step1 = cfg.data.edges_step1
    if not os.path.exists(nodes_step1) or not os.path.exists(edges_step1) or force:
        if not os.path.exists(cfg.data.raw_xml):
            raise FileNotFoundError(
                f"Không tìm thấy tệp DrugBank XML tại: {cfg.data.raw_xml}. "
                "Vui lòng tải 'full database.xml' vào thư mục data/raw/ hoặc sử dụng dữ liệu đã xử lý sẵn."
            )
        logger.info("Trích xuất dữ liệu từ DrugBank XML...")
        parser = DrugBankParser(cfg.data.raw_xml)
        parser.parse()
        parser.save_to_csv(nodes_step1, edges_step1)
    else:
        logger.info(f"Đã có file {nodes_step1} và {edges_step1}. Bỏ qua trích xuất XML.")

    # 2. Filter & Standardize Network
    nodes_clean = cfg.data.nodes_clean
    edges_clean = cfg.data.edges_clean
    if not os.path.exists(nodes_clean) or not os.path.exists(edges_clean) or force:
        logger.info("Làm sạch và chuẩn hóa mạng DDI...")
        filter_net = DDI_NetworkFilter(nodes_step1, edges_step1)
        filter_net.filter_and_standardize()
        filter_net.save_to_csv(nodes_clean, edges_clean)
    else:
        logger.info(f"Đã có file {nodes_clean} và {edges_clean}. Bỏ qua bước làm sạch.")

    # 3. Map SIDER Side Effects
    nodes_sider = cfg.data.nodes_sider
    nodes_final = cfg.data.nodes_final
    if not os.path.exists(nodes_final) or force:
        logger.info("Ánh xạ tác dụng phụ từ SIDER...")
        mapper = SIDER_DrugMapper(nodes_clean, cfg.data.sider_tsv, cache_dir=cfg.data.processed_dir)
        mapper.load_and_map_drugs()
        mapper.load_sider_data()
        mapper.process_mapping_to_list()
        mapper.save_output(nodes_sider, nodes_final)
    else:
        logger.info(f"Đã có file {nodes_final}. Bỏ qua bước SIDER.")

    logger.info("Hoàn tất chuẩn bị dữ liệu.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(args.config, args.force)
