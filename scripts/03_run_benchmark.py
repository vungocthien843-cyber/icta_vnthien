import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.seed import seed_everything
from src.models.baselines import BenchmarkPipeline
from src.evaluation.stat_tests import perform_statistical_tests

logger = setup_logger("Step03_Benchmark")


def run(config_path: str = "configs/default.yaml", output_dir: str = None):
    cfg = load_config(config_path)
    seed_everything(cfg.experiment.seed)

    out_dir = output_dir or os.path.join(cfg.output.results_dir, "tables")
    os.makedirs(out_dir, exist_ok=True)

    logger.info(f"Đọc dữ liệu huấn luyện từ: {cfg.data.train_csv}")
    if not os.path.exists(cfg.data.train_csv):
        raise FileNotFoundError(f"Chưa có file train_data.csv tại {cfg.data.train_csv}. Vui lòng chạy bước 02 trước.")

    train_df = pd.read_csv(cfg.data.train_csv)
    bench = BenchmarkPipeline(
        random_state=cfg.experiment.seed,
        n_splits=cfg.evaluation.cv_splits,
        use_gpu=cfg.experiment.use_gpu
    )

    results_df = bench.run_cv(train_df)
    
    # In ra màn hình console
    print("\n--- Kết quả Benchmark 5-Fold Cross Validation ---")
    print(results_df.to_string(index=False))

    # Lưu CSV
    csv_path = os.path.join(out_dir, "benchmark_cv_results.csv")
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Đã lưu kết quả CSV tại: {csv_path}")

    # Kiểm định thống kê
    if bench.f1_per_fold:
        stat_f1 = perform_statistical_tests(bench.f1_per_fold, metric_name="F1")
        if not stat_f1.empty:
            stat_csv = os.path.join(out_dir, "stat_tests_f1.csv")
            stat_f1.to_csv(stat_csv, index=False)
            logger.info(f"Đã lưu kiểm định thống kê tại: {stat_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    run(args.config, args.out_dir)
