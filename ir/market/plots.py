# -*- coding: utf-8 -*-
"""
ir/market/plots.py

Fonctions de plotting réutilisables dans notebooks.
But : garder le notebook lisible, en centralisant l'affichage.

Fonctions proposées :
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
    """
    Trace la courbe de marché :
    - à gauche : la courbe des discount factors P(0,t)
    - à droite : la courbe des forwards instantanés f(0,t)

    Paramètres
    ----------
    curve : objet Curve-like
    t_max : float
        Horizon maximal en années (par défaut 30 ans).
    n : int
        Nombre de points de discrétisation pour les courbes.
    title_prefix : str
        Préfixe ajouté au titre 
    """
    # Grille de temps
    t = np.linspace(0.01, float(t_max), int(n))

    # 2 panneaux côte à côte : DF et forward instantané
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # --- Discount curve ---
    axes[0].plot(t, curve.discount(t), linewidth=2)
    axes[0].set_title(f"{title_prefix} Courbe d'actualisation", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Temps (Années)")
    axes[0].set_ylabel("Facteur d'actualisation")
    axes[0].grid(alpha=0.3)

    # --- Instantaneous forward curve ---
    axes[1].plot(t, curve.inst_forward_rate(t), linewidth=2)
    axes[1].set_title(f"{title_prefix} Taux forward instantané", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Temps (Années)")
    axes[1].set_ylabel("Taux")
    axes[1].grid(alpha=0.3)

    # Ajustements visuels
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
    """
    Trace les prix (marché vs modèle) en fonction de l'expiry, pour plusieurs tenors.

    Usage typique :
    - df contient un ensemble de swaptions ATM
    - on veut comparer la term structure des prix pour des tenors fixes (5Y, 10Y, 20Y, 30Y)

    Paramètres
    ----------
    df : pd.DataFrame
        Doit contenir :
          - tenor_col (ex: "Tenor")
          - x_col (ex: "Expiry")
          - mkt_col (prix marché)
          - model_col (prix modèle)
    tenors : list[float] | None
        Tenors à afficher. Si None, valeurs par défaut [5, 10, 20, 30].
    tenor_col : str
        Nom de la colonne identifiant le tenor.
    x_col : str
        Nom de la colonne en abscisse (souvent Expiry).
    mkt_col : str
        Nom de la colonne prix marché.
    model_col : str
        Nom de la colonne prix modèle.
    ylabel : str
        Label axe Y (ex: "Forward Premium" si tu compares PV/DF(T0)).
    title : str
        Titre générique (complété par le tenor).
    """
    # Tenors par défaut si non fournis
    if tenors is None:
        tenors = [5.0, 10.0, 20.0, 30.0]

    # Layout 2x2 adapté au cas standard de 4 tenors
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    # Boucle sur les tenors à afficher
    for idx, tenor in enumerate(tenors):
        ax = axes[idx]
        dft = df[df[tenor_col] == tenor].copy()
        if len(dft) == 0:
            ax.set_axis_off()
            continue
        dft = dft.sort_values(x_col)

        # Courbe marché vs modèle
        ax.plot(dft[x_col], dft[mkt_col], linewidth=2, label="Marché")
        ax.plot(dft[x_col], dft[model_col], linestyle="--", linewidth=2, label="Modèle")

        # Mise en forme
        ax.set_title(f"{title} (ATM, {int(tenor)}Y Tenor)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Maturité (Années)")
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
    """
    Trace les volatilités normales (marché vs modèle) en fonction de l'expiry,
    pour plusieurs tenors.

    Usage typique :
    - df contient des vols implicites en bps (Bachelier) pour des swaptions ATM
    - on veut vérifier que le modèle reproduit bien la term structure des vols

    Paramètres
    ----------
    df : pd.DataFrame
        Doit contenir :
          - tenor_col, x_col
          - mkt_col : volatilité marché (en bps)
          - model_col : volatilité modèle (en bps)
    tenors : list[float] | None
        Tenors à afficher. Si None, valeurs par défaut [5, 10, 20, 30].
    tenor_col : str
        Colonne tenor.
    x_col : str
        Colonne expiry (abscisse).
    mkt_col : str
        Colonne vol marché.
    model_col : str
        Colonne vol modèle.
    ylabel : str
        Label axe Y.
    title : str
        Titre générique (complété par le tenor).
    """
    if tenors is None:
        tenors = [5.0, 10.0, 20.0, 30.0]

    # Layout 2x2 adapté au cas standard de 4 tenors
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for idx, tenor in enumerate(tenors):
        ax = axes[idx]
        dft = df[df[tenor_col] == tenor].copy()
        if len(dft) == 0:
            ax.set_axis_off()
            continue
        dft = dft.sort_values(x_col)

        ax.plot(dft[x_col], dft[mkt_col], linewidth=2, label="Marché")
        ax.plot(dft[x_col], dft[model_col], linestyle="--", linewidth=2, label="Modèle")

        ax.set_title(f"{title} (ATM, {int(tenor)}Y Tenor)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Maturité (Années)")
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
    Trace des smiles (caplets OTM) en regroupant par expiry.

    Données attendues
    -----------------
    df doit contenir au moins :
      - expiry_col : maturité (Expiry)
      - x_col : moneyness (souvent en % ou en bps selon le format)
      - mkt_col : vol implicite marché (en bps)
      - model_col : vol implicite modèle (même unité que mkt_col)

    Paramètres
    ----------
    expiry_col : str
        Colonne identifiant l'expiry.
    x_col : str
        Colonne en abscisse (moneyness).
    mkt_col : str
        Colonne vol marché.
    model_col : str
        Colonne vol modèle.
    n_panels : int
        Nombre de panneaux (expiries) à afficher, en choisissant des expiries "réparties".
    title_prefix : str
        Préfixe de titre (utile si plusieurs types de smiles).
    """
    # Liste triée des expiries disponibles
    expiries = np.sort(df[expiry_col].dropna().unique())
    if len(expiries) == 0:
        raise ValueError("Pas de maturités trouvés pour le graphique du smile.")

    n_panels = int(n_panels)

    if len(expiries) >= n_panels:
        exp_indices = np.linspace(0, len(expiries) - 1, n_panels, dtype=int)
    else:
        exp_indices = np.arange(len(expiries))

    # Layout 
    if len(exp_indices) <= 4:
        nrows, ncols = 2, 2
    else:
        ncols = int(np.ceil(np.sqrt(len(exp_indices))))
        nrows = int(np.ceil(len(exp_indices) / ncols))

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8))
    axs = np.array(axs).reshape(-1)

    # Tracé panneau par panneau
    for k, idx in enumerate(exp_indices):
        ax = axs[k]
        expiry = expiries[idx]
        mask = df[expiry_col] == expiry
        dft = df[mask].sort_values(x_col)

        # Tracé marché vs modèle
        ax.plot(dft[x_col], dft[mkt_col], linewidth=2, label="Smile Marché")
        ax.plot(dft[x_col], dft[model_col], linestyle="--", linewidth=2, label="Smile Modèle")

        # Mise en forme
        ax.set_title(f"{title_prefix} | Maturité: {expiry:.2f}Y", fontsize=12, fontweight="bold")
        ax.set_xlabel("Moneyness (%)")
        ax.set_ylabel("Vol Impli (Bps)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    for j in range(len(exp_indices), len(axs)):
        axs[j].set_axis_off()

    plt.tight_layout()
    plt.show()
