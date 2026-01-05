# -*- coding: utf-8 -*-
"""
ir/market/plots.py

Fonctions de plotting réutilisables dans notebooks.
But: garder le notebook lisible.

- plot_curve(curve)
- plot_prices_by_tenor(df, ...)
- plot_vols_by_tenor(df, ...)
- plot_smile_by_expiry(df, ...)  (caplets OTM)
"""

from __future__ import annotations

from typing import Sequence, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_curve(curve, t_max: float = 30.0, n: int = 300, title_prefix: str = "Market") -> None:
    t = np.linspace(0.01, float(t_max), int(n))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(t, curve.discount(t), linewidth=2)
    axes[0].set_title(f"{title_prefix} Discount Curve", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Time (Years)")
    axes[0].set_ylabel("Discount Factor")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, curve.inst_forward_rate(t), linewidth=2)
    axes[1].set_title(f"{title_prefix} Instantaneous Forward Rate", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Time (Years)")
    axes[1].set_ylabel("Rate")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_prices_by_tenor(
    df: pd.DataFrame,
    tenors: Optional[Sequence[float]] = None,
    tenor_col: str = "Tenor",
    x_col: str = "Expiry",
    mkt_col: str = "Price",
    model_col: str = "Model_Price",
    ylabel: str = "Forward Premium",
    title: str = "Swaption Price Term Structure",
) -> None:
    if tenors is None:
        tenors = [5.0, 10.0, 20.0, 30.0]

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

        ax.set_title(f"{title} (ATM, {int(tenor)}Y Tenor)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Expiry (Years)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.show()


def plot_vols_by_tenor(
    df: pd.DataFrame,
    tenors: Optional[Sequence[float]] = None,
    tenor_col: str = "Tenor",
    x_col: str = "Expiry",
    mkt_col: str = "Volatility (Bps)",
    model_col: str = "Model_Vol (Bps)",
    ylabel: str = "Normal Vol (Bps)",
    title: str = "Swaption Vol Term Structure",
) -> None:
    if tenors is None:
        tenors = [5.0, 10.0, 20.0, 30.0]

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

        ax.set_title(f"{title} (ATM, {int(tenor)}Y Tenor)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Expiry (Years)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.show()


def plot_smile_by_expiry(
    df: pd.DataFrame,
    expiry_col: str = "Expiry",
    x_col: str = "Moneyness",
    mkt_col: str = "Volatility (Bps)",
    model_col: str = "Model Vol",
    n_panels: int = 4,
    title_prefix: str = "Caplet Smile",
) -> None:
    """
    df: DataFrame contenant au moins Expiry, Moneyness, Market vol, Model vol.
    """
    expiries = np.sort(df[expiry_col].dropna().unique())
    if len(expiries) == 0:
        raise ValueError("No expiries found for smile plot.")

    # pick n_panels expiries spread out
    n_panels = int(n_panels)
    if len(expiries) >= n_panels:
        exp_indices = np.linspace(0, len(expiries) - 1, n_panels, dtype=int)
    else:
        exp_indices = np.arange(len(expiries))

    # grid layout: for 4 -> 2x2 ; else approximate square
    if len(exp_indices) <= 4:
        nrows, ncols = 2, 2
    else:
        ncols = int(np.ceil(np.sqrt(len(exp_indices))))
        nrows = int(np.ceil(len(exp_indices) / ncols))

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8))
    axs = np.array(axs).reshape(-1)

    for k, idx in enumerate(exp_indices):
        ax = axs[k]
        expiry = expiries[idx]
        mask = df[expiry_col] == expiry
        dft = df[mask].sort_values(x_col)

        ax.plot(dft[x_col], dft[mkt_col], linewidth=2, label="Market Smile")
        ax.plot(dft[x_col], dft[model_col], linestyle="--", linewidth=2, label="Model Smile")

        ax.set_title(f"{title_prefix} | Expiry: {expiry:.2f}Y", fontsize=12, fontweight="bold")
        ax.set_xlabel("Moneyness (%)")
        ax.set_ylabel("Implied Vol (Bps)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    # hide remaining axes
    for j in range(len(exp_indices), len(axs)):
        axs[j].set_axis_off()

    plt.tight_layout()
    plt.show()
