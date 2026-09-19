import os
from typing import Any
from ..utils.logger import setup_logger

logger = setup_logger("Export")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


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
