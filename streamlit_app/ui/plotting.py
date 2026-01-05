from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def _human_money(x, pos=None) -> str:
    x = float(x)
    ax = abs(x)
    if ax >= 1e9:
        return f"{x/1e9:.1f}B"
    if ax >= 1e6:
        return f"{x/1e6:.1f}M"
    if ax >= 1e3:
        return f"{x/1e3:.1f}k"
    return f"{x:.0f}"


def fig_curve(curve, t_max: float = 30.0, n: int = 300, title_prefix: str = "Market"):
    t = np.linspace(0.01, float(t_max), int(n))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(t, curve.discount(t), linewidth=2)
    axes[0].set_title(f"{title_prefix} Discount Curve", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Time (Years)")
    axes[0].set_ylabel("Discount Factor")
    axes[0].grid(alpha=0.25)

    axes[1].plot(t, curve.inst_forward_rate(t), linewidth=2)
    axes[1].set_title(f"{title_prefix} Instantaneous Forward Rate", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Time (Years)")
    axes[1].set_ylabel("Rate")
    axes[1].grid(alpha=0.25)

    plt.tight_layout()
    return fig


def fig_prices_by_tenor(
    df: pd.DataFrame,
    *,
    tenors=(5.0, 10.0, 20.0, 30.0),
    tenor_col="Tenor",
    x_col="Expiry",
    mkt_col="Price",
    model_col="Model_Price",
    ylabel="Forward Premium",
    title="Swaption Price Term Structure",
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for idx, tenor in enumerate(tenors):
        ax = axes[idx]
        dft = df[df[tenor_col] == tenor].copy()
        if len(dft) == 0:
            ax.set_axis_off()
            continue
        dft = dft.sort_values(x_col)

        ax.plot(dft[x_col], dft[mkt_col], linewidth=2, label="Market")
        ax.plot(dft[x_col], dft[model_col], linestyle="--", linewidth=2, label="Model")
        ax.set_title(f"{title} (ATM, {int(tenor)}Y)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Expiry (Years)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend()

    plt.tight_layout()
    return fig


def fig_vols_by_tenor(
    df: pd.DataFrame,
    *,
    tenors=(5.0, 10.0, 20.0, 30.0),
    tenor_col="Tenor",
    x_col="Expiry",
    mkt_col="Market_Vol (Bps)",
    model_col="Model_Vol (Bps)",
    ylabel="Normal Vol (Bps)",
    title="Swaption Normal Vol Term Structure",
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for idx, tenor in enumerate(tenors):
        ax = axes[idx]
        dft = df[df[tenor_col] == tenor].copy()
        if len(dft) == 0:
            ax.set_axis_off()
            continue
        dft = dft.sort_values(x_col)

        ax.plot(dft[x_col], dft[mkt_col], linewidth=2, label="Market")
        ax.plot(dft[x_col], dft[model_col], linestyle="--", linewidth=2, label="Model")
        ax.set_title(f"{title} (ATM, {int(tenor)}Y)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Expiry (Years)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend()

    plt.tight_layout()
    return fig


def fig_pfe(grid, pfe, epe=None, q: float = 0.95, title: str = "PFE profile", subtitle: str | None = None):
    grid = np.asarray(grid, dtype=float)
    pfe = np.asarray(pfe, dtype=float)
    epe = None if epe is None else np.asarray(epe, dtype=float)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(grid, pfe, marker="o", markersize=3.5, linewidth=2.0, label=f"PFE ({int(q*100)}%)")
    ax.fill_between(grid, 0.0, pfe, alpha=0.12)

    if epe is not None:
        ax.plot(grid, epe, marker="s", markersize=3.2, linewidth=1.8, linestyle="--", label="EPE")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    if subtitle:
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=10, alpha=0.9, va="bottom")

    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Exposure")
    ax.yaxis.set_major_formatter(FuncFormatter(_human_money))
    ax.grid(alpha=0.25)
    ax.set_xlim(grid.min(), grid.max())
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="upper right")

    plt.tight_layout()
    return fig
