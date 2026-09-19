import os
from typing import Optional, Any
import pandas as pd
from ..utils.logger import setup_logger

logger = setup_logger("Export")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def escape_latex(text: Any) -> str:
    """Escape các ký tự đặc biệt trong LaTeX (như _, %) và chuyển ± thành $\pm$."""
    if pd.isna(text) or str(text).strip().lower() in ("nan", "none", "<na>"):
        return "—"
    s = str(text)
    if s.startswith("$") and s.endswith("$"):
        return s
    s = s.replace("_", r"\_").replace("%", r"\%")
    if "±" in s:
        s = s.replace("±", r"$\pm$")
    return s


def export_latex_table(df: pd.DataFrame, output_path: str,
                       caption: str = "", label: str = "",
                       bold_best_col: Optional[str] = None) -> str:
    """Xuất DataFrame ra bảng LaTeX định dạng booktabs."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean = df.copy()

    # Định dạng các cột số và xử lý NaN
    for col in df_clean.columns:
        is_p_val = any(p_sub in str(col).lower() for p_sub in ['_p', 'p_val', 'p-val'])
        if pd.api.types.is_float_dtype(df_clean[col]):
            fmt = "{:.4f}" if is_p_val else "{:.2f}"
            df_clean[col] = df_clean[col].map(lambda x: fmt.format(x) if pd.notna(x) else "—")
        else:
            df_clean[col] = df_clean[col].map(lambda x: "—" if pd.isna(x) or str(x).lower() == 'nan' else str(x))

    col_align = "l" + "c" * (len(df_clean.columns) - 1)
    escaped_headers = [escape_latex(col) for col in df_clean.columns]
    
    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}" if caption else "",
        f"\\label{{{label}}}" if label else "",
        f"\\begin{{tabular}}{{{col_align}}}",
        r"\toprule",
        " & ".join(escaped_headers) + r" \\",
        r"\midrule"
    ]

    for _, row in df_clean.iterrows():
        row_vals = [escape_latex(val) for val in row.values]
        latex_lines.append(" & ".join(row_vals) + r" \\")

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])

    latex_str = "\n".join([line for line in latex_lines if line])
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex_str)
    
    logger.info(f"Đã xuất bảng LaTeX tại: {output_path}")
    return latex_str


def save_plot(fig: Any, output_path: str, dpi: int = 300) -> None:
    """
    Lưu đồ thị thành file vector (PDF/SVG) hoặc bitmap (PNG 300 DPI) chất lượng cao.
    """
    if not HAS_MATPLOTLIB:
        logger.warning("Chưa cài đặt 'matplotlib'. Không thể lưu biểu đồ (pip install matplotlib).")
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    logger.info(f"Đã lưu biểu đồ tại: {output_path}")
    plt.close(fig)
