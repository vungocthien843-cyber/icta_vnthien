import os
import sys
import shutil
import argparse
import importlib
from datetime import datetime

# Đảm bảo workspace root luôn có trong sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.seed import seed_everything

# Import các bước kịch bản thông qua importlib (do tên file bắt đầu bằng số)
step1 = importlib.import_module("scripts.01_prepare_data")
step2 = importlib.import_module("scripts.02_build_features")
step3 = importlib.import_module("scripts.03_run_benchmark")
step4 = importlib.import_module("scripts.04_run_ablation")
step5 = importlib.import_module("scripts.05_run_ensemble")

logger = setup_logger("MainRunner")


def sync_to_latest(run_tables_dir: str, base_results_dir: str):
    """Copy kết quả từ lần chạy hiện tại sang results/latest/tables/."""
    latest_tables_dir = os.path.join(base_results_dir, "latest", "tables")
    os.makedirs(latest_tables_dir, exist_ok=True)
    
    if os.path.exists(run_tables_dir):
        for fname in os.listdir(run_tables_dir):
            src_f = os.path.join(run_tables_dir, fname)
            dst_f = os.path.join(latest_tables_dir, fname)
            if os.path.isfile(src_f):
                shutil.copy2(src_f, dst_f)
        logger.info(f"Đã cập nhật kết quả sang: {latest_tables_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Drug-Drug Interaction Prediction via SHAP and Ensemble Learning"
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Đường dẫn file cấu hình YAML")
    parser.add_argument(
        "--stage", type=str, default="all",
        choices=["preprocess", "features", "benchmark", "ablation", "ensemble", "all"],
        help="Giai đoạn thực thi (mặc định: all)"
    )
    parser.add_argument("--exp-name", type=str, default=None, help="Tên lần chạy thử nghiệm")
    parser.add_argument("--force", action="store_true", help="Tính lại dữ liệu kể cả khi đã có cache")
    args = parser.parse_args()

    config_path = args.config
    cfg = load_config(config_path)

    seed_everything(cfg.experiment.seed)

    exp_name = args.exp_name or cfg.experiment.name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg.output.results_dir, "runs", f"{timestamp}_{exp_name}")
    run_tables_dir = os.path.join(run_dir, "tables")
    os.makedirs(run_tables_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config_used.yaml"), "w", encoding="utf-8") as f:
        with open(config_path, "r", encoding="utf-8") as sf:
            f.write(sf.read())

    logger.info(f"Bắt đầu chạy pipeline (stage: {args.stage}, config: {config_path}, seed: {cfg.experiment.seed})")

    # 1. Giai đoạn Preprocess
    if args.stage in ["preprocess", "all"]:
        logger.info("Bước 1: Tiền xử lý dữ liệu thô (DrugBank XML & SIDER)...")
        step1.run(config_path=config_path, force=args.force)

    # 2. Giai đoạn Features
    if args.stage in ["features", "all"]:
        logger.info("Bước 2: Trích xuất đặc trưng (Louvain G_train, Topology, Semantic)...")
        step2.run(config_path=config_path, force=args.force)

    # 3. Giai đoạn Benchmark
    if args.stage in ["benchmark", "all"]:
        logger.info("Bước 3: Huấn luyện và đánh giá 5-Fold Cross Validation các mô hình Baseline...")
        step3.run(config_path=config_path, output_dir=run_tables_dir)

    # 4. Giai đoạn Ablation & SHAP
    if args.stage in ["ablation", "all"]:
        logger.info("Bước 4: Phân tích độ quan trọng đặc trưng (Leave-One-Out, Group, SHAP)...")
        step4.run(config_path=config_path, output_dir=run_tables_dir)

    # 5. Giai đoạn Ensemble
    if args.stage in ["ensemble", "all"]:
        logger.info("Bước 5: Huấn luyện và đánh giá mô hình kết hợp (Voting, Stacking)...")
        step5.run(config_path=config_path, output_dir=run_tables_dir)

    # Đồng bộ sang kết quả latest
    sync_to_latest(run_tables_dir, cfg.output.results_dir)

    logger.info(f"Hoàn thành. Kết quả lưu tại: {run_tables_dir}")
    logger.info(f"Bản tổng hợp mới nhất: {os.path.join(cfg.output.results_dir, 'latest', 'tables')}")


if __name__ == "__main__":
    main()
