import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.seed import seed_everything
from src.evaluation.explainability import AblationAndSHAP

logger = setup_logger("Step04_AblationSHAP")


def run(config_path: str = "configs/default.yaml", output_dir: str = None):
    cfg = load_config(config_path)
    seed_everything(cfg.experiment.seed)

    out_dir = output_dir or os.path.join(cfg.output.results_dir, "tables")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(cfg.data.train_csv) or not os.path.exists(cfg.data.test_csv):
        raise FileNotFoundError("Chưa tìm thấy train_data.csv hoặc test_data.csv. Hãy chạy bước 02 trước.")

    train_df = pd.read_csv(cfg.data.train_csv)
    test_df = pd.read_csv(cfg.data.test_csv)

    ablation = AblationAndSHAP(train_df, test_df, seed=cfg.experiment.seed)

    # 1. Leave-One-Out
    loo_df = ablation.leave_one_out()
    print("\n--- Kết quả Leave-One-Out ---")
    print(loo_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    loo_csv = os.path.join(out_dir, "ablation_leave_one_out.csv")
    loo_df.to_csv(loo_csv, index=False, float_format="%.2f")
    logger.info(f"Đã lưu kết quả Leave-One-Out tại: {loo_csv}")

    # 2. Group Ablation
    grp_df = ablation.group_ablation()
    print("\n--- Kết quả Group Ablation ---")
    print(grp_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    grp_csv = os.path.join(out_dir, "ablation_groups.csv")
    grp_df.to_csv(grp_csv, index=False, float_format="%.2f")
    logger.info(f"Đã lưu kết quả Group Ablation tại: {grp_csv}")

    # 3. SHAP Cumulative
    shap_df = ablation.shap_cumulative(sample_size=cfg.evaluation.shap_sample_size)
    if shap_df is not None:
        print("\n--- Kết quả SHAP Cumulative Addition ---")
        print(shap_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        shap_csv = os.path.join(out_dir, "shap_cumulative.csv")
        shap_df.to_csv(shap_csv, index=False, float_format="%.2f")
        logger.info(f"Đã lưu kết quả SHAP Cumulative tại: {shap_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    run(args.config, args.out_dir)
