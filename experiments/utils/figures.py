"""Publication-quality matplotlib defaults + helpers."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def setup_matplotlib() -> None:
    """Set publication defaults. Idempotent."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    })


def savefig(fig, path: str | os.PathLike, also_png: bool = True) -> tuple[Path, Optional[Path]]:
    """Save figure as PDF + (optionally) PNG."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = p.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    png_path = None
    if also_png:
        png_path = p.with_suffix(".png")
        fig.savefig(png_path, bbox_inches="tight", dpi=300)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return pdf_path, png_path


# Color palette used across experiments — readable, colorblind-friendly.
PALETTE = [
    "#2196F3", "#FF9800", "#4CAF50", "#E91E63",
    "#9C27B0", "#00BCD4", "#FFC107", "#795548",
]
