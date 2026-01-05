# -*- coding: utf-8 -*-
"""
pfe_plot.py

Plot helpers for PFE/EPE profiles (matplotlib).
Extracted from your notebook so you can import it later.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def _human_money(x, pos=None) -> str:
    """Format 1200 -> 1.2k, 1200000 -> 1.2M."""
    x = float(x)
    ax = abs(x)
    if ax >= 1e9:
        return f"{x/1e9:.1f}B"
    if ax >= 1e6:
        return f"{x/1e6:.1f}M"
    if ax >= 1e3:
        return f"{x/1e3:.1f}k"
    return f"{x:.0f}"


def plot_pfe_profile(
    grid,
    pfe,
    epe=None,
    q: float = 0.95,
    title: str = "PFE profile",
    subtitle: str | None = None,
    xlabel: str = "Time (years)",
    ylabel: str = "Exposure",
    show_fill: bool = True,
    annotate_peak: bool = True,
    savepath: str | None = None,
):
    """
    Plot a clean PFE profile (and optionally EPE).

    Parameters
    ----------
    grid : array-like
        Time grid in years.
    pfe : array-like
        PFE values (same length as grid).
    epe : array-like, optional
        EPE values (same length as grid).
    q : float
        Quantile used for PFE (e.g., 0.95, 0.975, 0.99).
    subtitle : str, optional
        Additional context line (swap details, notional, K, model params...).
    show_fill : bool
        If True, fill under PFE curve (subtle).
    annotate_peak : bool
        If True, annotate max PFE point.
    savepath : str, optional
        If provided, saves figure as PNG.

    Returns
    -------
    (fig, ax)
    """
    grid = np.asarray(grid, dtype=float)
    pfe = np.asarray(pfe, dtype=float)
    if epe is not None:
        epe = np.asarray(epe, dtype=float)

    fig, ax = plt.subplots(figsize=(11, 5.5))

    label_pfe = f"PFE ({int(q*100)}%)"
    ax.plot(grid, pfe, marker="o", markersize=3.5, linewidth=2.0, label=label_pfe)

    if show_fill:
        ax.fill_between(grid, 0.0, pfe, alpha=0.12)

    if epe is not None:
        ax.plot(grid, epe, marker="s", markersize=3.2, linewidth=1.8, linestyle="--", label="EPE")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    if subtitle:
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=10, alpha=0.9, va="bottom")

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.yaxis.set_major_formatter(FuncFormatter(_human_money))
    ax.grid(True, alpha=0.25)

    ax.set_xlim(grid.min(), grid.max())
    ax.set_ylim(bottom=0.0)

    if annotate_peak and len(pfe) > 0:
        i_max = int(np.argmax(pfe))
        t_max = grid[i_max]
        pfe_max = pfe[i_max]
        ax.scatter([t_max], [pfe_max], s=45, zorder=5)
        ax.annotate(
            f"Peak: {pfe_max:,.0f} at {t_max:.2f}y",
            xy=(t_max, pfe_max),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=10,
            arrowprops=dict(arrowstyle="->", lw=1.0, alpha=0.8),
        )

    ax.legend(frameon=True, fontsize=10, loc="upper right")
    plt.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")

    plt.show()
    return fig, ax
