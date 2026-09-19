# Improving Drug–Drug Interaction Prediction via SHAP-Based Feature Selection and Ensemble Learning

Dự án dự đoán tương tác thuốc–thuốc (Drug–Drug Interaction, DDI) từ cơ sở dữ liệu DrugBank 5.0 và SIDER.

Pipeline kết hợp 4 đặc trưng tương đồng y sinh / cấu trúc hóa học (ATC, MeSH, ADE, SMILES) và 8 đặc trưng cấu trúc mạng đồ thị (Louvain community và neighborhood). Hệ thống đánh giá qua 8 mô hình ML baseline, phân tích độ quan trọng đặc trưng (Leave-One-Out, SHAP) và mô hình kết hợp Ensemble (Voting, Stacking).

---

## 1. Luồng xử lý

```text
DrugBank XML ──► Làm sạch nodes & edges ──► Ghép SIDER (ADE)
  │
  ├──► 4 Đặc trưng Ngữ nghĩa & Hóa học (ATC, MeSH, ADE, SMILES)
  │
  └──► Đồ thị DDI ──► Tách cạnh dương (Train/Test) ──► Louvain trên G_train
                         │
                         └──► 8 Đặc trưng Cấu trúc (Neighborhood + Community)
                                │
                                └──► Ghép 12 đặc trưng (Final_Dataset_DDI.csv)
                                       │
                                       ├──► 5-Fold Cross Validation & Statistical Tests
                                       ├──► Leave-One-Out & SHAP Feature Ablation
                                       └──► Ensemble Learning (Voting, Stacking)
                                              │
                                              └──► Xuất Bảng Kết Quả Định Dạng CSV
```

---

## 2. Cấu trúc Thư mục

```text
Restruct_code_ICTA/
├── configs/                          # File cấu hình tham số thực nghiệm (YAML)
│   └── default.yaml                  # Cấu hình mặc định
│
├── data/                             # Thư mục dữ liệu
│   ├── raw/                          # Dữ liệu gốc (full database.xml, meddra_all_se.tsv)
│   └── processed/                    # Dữ liệu trung gian và ma trận đặc trưng
│
├── src/                              # Mã nguồn chính
│   ├── data/                         # Parser XML, bộ lọc mạng DDI, mapper SIDER, chia train/test
│   ├── features/                     # Sinh 4 semantic + 8 topology features
│   ├── models/                       # 8 ML baselines và Ensemble (hỗ trợ GPU cuML và CPU)
│   ├── evaluation/                   # Độ đo đánh giá, Paired t-test, Wilcoxon, SHAP
│   └── utils/                        # Đọc config, random seed, logger
│
├── scripts/                          # Script chạy từng bước độc lập
│   ├── 01_prepare_data.py            # Bước 1: Parse XML và ghép SIDER
│   ├── 02_build_features.py          # Bước 2: Sinh đặc trưng và dựng đồ thị G_train
│   ├── 03_run_benchmark.py           # Bước 3: Benchmark 5-Fold CV và kiểm định thống kê
│   ├── 04_run_ablation.py            # Bước 4: Ablation study và SHAP
│   └── 05_run_ensemble.py            # Bước 5: Voting và Stacking
│
├── results/                          # Kết quả đầu ra
│   ├── latest/tables/                # Bảng kết quả mới nhất định dạng CSV
│   ├── runs/                         # Lịch sử các lần chạy
│   └── logs/                         # File log
│
├── tests/                            # Unit tests
│   ├── test_leakage.py               # Kiểm tra G_train không chứa cạnh test
│   └── test_pipeline.py              # Kiểm tra tính toàn vẹn 12 đặc trưng và nhãn
│
├── main.py                           # CLI điều phối toàn bộ pipeline
├── pytest.ini                        # Cấu hình pytest
├── requirements.txt                  # Thư viện phụ thuộc pip
└── environment.yml                   # Cấu hình môi trường conda
```

---

## 3. Cài đặt Môi trường

Khuyến nghị sử dụng Python 3.10 hoặc 3.12:

```bash
# Cài đặt qua pip
pip install -r requirements.txt

# Hoặc cài đặt qua Conda
conda env create -f environment.yml
conda activate ddi_env
```

Lưu ý về phần cứng:
- Nếu môi trường có GPU NVIDIA và cài RAPIDS `cuml`, các mô hình RF, SVM, kNN, LR sẽ tự động chạy trên GPU.
- Trên môi trường CPU thông thường, hệ thống tự động sử dụng Scikit-learn.

---

## 4. Hướng dẫn Chạy

### 4.1. Chạy toàn bộ Pipeline

Chạy toàn bộ quy trình:

```bash
python main.py
```

Quy trình thực hiện:
1. Đọc dữ liệu đặc trưng tại `data/processed/`.
2. Đánh giá 5-Fold Cross Validation trên 8 mô hình ML baseline (báo cáo Mean $\pm$ Std).
3. Kiểm định thống kê Paired t-test và Wilcoxon signed-rank test.
4. Chạy Leave-One-Out, Group Feature Ablation và SHAP Cumulative Feature Addition.
5. Đánh giá mô hình kết hợp Ensemble (Hard Voting, Soft Voting, Stacking).
6. Xuất các bảng kết quả dạng `.csv` vào thư mục `results/latest/tables/`.

### 4.2. Chạy từng bước độc lập

Sử dụng tham số `--stage` của `main.py` hoặc gọi trực tiếp file trong `scripts/`:

```bash
# Bước 1: Tiền xử lý dữ liệu thô (cần full database.xml và meddra_all_se.tsv trong data/raw/)
python main.py --stage preprocess
# hoặc: python scripts/01_prepare_data.py

# Bước 2: Tính đặc trưng và phân chia Train/Test
python main.py --stage features
# hoặc: python scripts/02_build_features.py

# Bước 3: Benchmark 8 mô hình (5-Fold CV)
python main.py --stage benchmark
# hoặc: python scripts/03_run_benchmark.py

# Bước 4: Ablation study và SHAP
python main.py --stage ablation
# hoặc: python scripts/04_run_ablation.py

# Bước 5: Huấn luyện Ensemble
python main.py --stage ensemble
# hoặc: python scripts/05_run_ensemble.py
```

### 4.3. Chạy Unit Tests

Kiểm tra tính toàn vẹn dữ liệu và đảm bảo không có rò rỉ dữ liệu giữa tập train và test:

```bash
pytest -v
```

---

## 5. Quản lý Kết quả

1. **Dữ liệu trung gian (`data/processed/`)**:
   - Các ma trận đặc trưng và phân cụm Louvain được lưu lại sau khi tính. Nếu muốn tính lại từ đầu, thêm cờ `--force`.

2. **Lưu trữ kết quả (`results/`)**:
   - Mỗi lần chạy được lưu vào thư mục riêng gắn timestamp tại `results/runs/<timestamp>_<name>/`.
   - Kết quả mới nhất luôn được đồng bộ về `results/latest/tables/`:
     ```text
     results/latest/tables/
     ├── benchmark_cv_results.csv
     ├── stat_tests_f1.csv
     ├── ablation_leave_one_out.csv
     ├── ablation_groups.csv
     ├── shap_cumulative.csv
     └── ensemble_test_results.csv
     ```

Các file `.csv` có thể mở trực tiếp bằng Microsoft Excel, Google Sheets, hoặc các phần mềm bảng tính và data science thông dụng.

---

## 6. Lưu ý

1. **Bảo vệ rò rỉ dữ liệu (Data Leakage)**: Thuật toán phân cụm Louvain và các đặc trưng đồ thị (topology) chỉ được tính trên $G_{train}$, không chứa cạnh test.
2. **Cạnh dương**: Giữ cố định phân chia cạnh dương từ bước phân cụm sang tập train/test cuối cùng qua `pos_edge_split.pkl`.
3. **Reproducibility**: Thiết lập seed cố định (`seed=42`) trong `configs/default.yaml` cho toàn bộ các bước ngẫu nhiên.
4. **DrugBank XML**: File `full database.xml` có dung lượng lớn và bản quyền hạn chế nên không được commit lên git (đã được cấu hình trong `.gitignore`).

---

## 7. Trích dẫn

```bibtex
@article{ddi_shap_ensemble_2026,
  title={Improving Drug--Drug Interaction Prediction via SHAP-Based Feature Selection and Ensemble Learning},
  author={Vu Ngoc Thien and Collaborators},
  journal={Conference / Journal Name},
  year={2026}
}
```
