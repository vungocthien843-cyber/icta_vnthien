import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.seed import seed_everything
from src.models.ensemble import DDIEnsembleTrainer
from src.evaluation.export import export_latex_table

logger = setup_logger("Step05_Ensemble")


def run(config_path: str = "configs/default.yaml", output_dir: str = None):
    cfg = load_config(config_path)
    seed_everything(cfg.experiment.seed)

    out_dir = output_dir or os.path.join(cfg.output.results_dir, "tables")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(cfg.data.train_csv) or not os.path.exists(cfg.data.test_csv):
        raise FileNotFoundError("Chưa tìm thấy train_data.csv hoặc test_data.csv. Hãy chạy bước 02 trước.")

    train_df = pd.read_csv(cfg.data.train_csv)
    test_df = pd.read_csv(cfg.data.test_csv)

    trainer = DDIEnsembleTrainer(seed=cfg.experiment.seed, use_gpu=cfg.experiment.use_gpu)
    ens_df = trainer.run(
        train_df, test_df,
        cv=cfg.evaluation.stacking_cv,
        passthrough=cfg.evaluation.stacking_passthrough
    )

    print("\n--- Kết quả Ensemble trên Test Holdout ---")
    print(ens_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    csv_path = os.path.join(out_dir, "ensemble_test_results.csv")
    tex_path = os.path.join(out_dir, "ensemble_test_results.tex")
    ens_df.to_csv(csv_path, index=False, float_format="%.2f")
    export_latex_table(
        ens_df, tex_path,
        caption="Comparison between individual base learners and ensemble strategies (Voting, Stacking) on DDI test holdout.",
        label="tab:ensemble_results"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    run(args.config, args.out_dir)
