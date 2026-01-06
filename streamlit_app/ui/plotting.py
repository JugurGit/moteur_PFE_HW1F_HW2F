from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# -----------------------------------------------------------------------------
# But du module
# -----------------------------------------------------------------------------
# Ce fichier regroupe des fonctions "fig_*" qui construisent des figures Matplotlib
# prêtes à être affichées dans Streamlit via:
#   st.pyplot(fig, clear_figure=True)
# -----------------------------------------------------------------------------


def _human_money(x, pos=None) -> str:
    """
    Formatte un nombre en "human readable" (k, M, B).

    Exemple:
      1200    -> "1.2k"
      1200000 -> "1.2M"
    """
    x = float(x)
    ax = abs(x)
    if ax >= 1e9:
        return f"{x/1e9:.1f}B"
    if ax >= 1e6:
        return f"{x/1e6:.1f}M"
    if ax >= 1e3:
        return f"{x/1e3:.1f}k"
    return f"{x:.0f}"


def fig_curve(curve, t_max: float = 30.0, n: int = 300, title_prefix: str = "Marché"):
    """
    Figure 2 panneaux pour visualiser :
      - la courbe d'actualisation P(0,t)
      - le forward instantané f(0,t)

    Paramètres
    ----------
    curve:
        Objet de type Curve (ou compatible) exposant:
          - discount(t)
          - inst_forward_rate(t)
    t_max:
        Horizon max en années (ex: 30y)
    n:
        Nombre de points sur la grille pour l'affichage
    title_prefix:
        Préfixe du titre (ex: "Marché", "Modèle", ...)

    Returns
    -------
    fig : matplotlib.figure.Figure
        La figure Matplotlib.
    """
    t = np.linspace(0.01, float(t_max), int(n))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Panneau 1 : discount factors
    axes[0].plot(t, curve.discount(t), linewidth=2)
    axes[0].set_title(f"{title_prefix} Courbe d'actualisation", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Temps (Années)")
    axes[0].set_ylabel("Facteur d'actualisation")
    axes[0].grid(alpha=0.25)

    # Panneau 2 : forward instantané
    axes[1].plot(t, curve.inst_forward_rate(t), linewidth=2)
    axes[1].set_title(f"{title_prefix} Taux forward instantané", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Temps (Années)")
    axes[1].set_ylabel("Taux")
    axes[1].grid(alpha=0.25)

    plt.tight_layout()
    return fig


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _pro_axes_style(ax):
    ax.grid(True, alpha=0.18, linewidth=0.8)
    # alléger les bordures
    ax.spines["top"].set_alpha(0.25)
    ax.spines["right"].set_alpha(0.25)
    ax.spines["left"].set_alpha(0.35)
    ax.spines["bottom"].set_alpha(0.35)
    ax.tick_params(axis="both", labelsize=9)


def fig_prices_by_tenor(
    df: pd.DataFrame,
    *,
    tenors=(5.0, 10.0, 20.0, 30.0),
    tenor_col="Tenor",
    x_col="Expiry",
    mkt_col="Price",
    model_col="Model_Price",
    ylabel="Forward Premium",
    title="Courbe (Prix swaptions)",
    subtitle: str | None = None,
    show_error_band: bool = False,
):

    # --- layout
    n_panels = min(len(tenors), 4)
    if n_panels <= 0:
        raise ValueError("Doit contenir au moins deux tenors.")

    fig, axes = plt.subplots(
        2, 2,
        figsize=(12.5, 8.2),
        sharex=True,
        constrained_layout=False
    )
    axes = axes.flatten()

    handles_global = None
    labels_global = None

    for idx, tenor in enumerate(tenors):
        if idx >= len(axes):
            break

        ax = axes[idx]
        dft = df[df[tenor_col] == tenor].copy()
        if dft.empty:
            ax.set_axis_off()
            continue

        dft = dft.sort_values(x_col)

        x = dft[x_col].to_numpy(dtype=float)
        y_mkt = dft[mkt_col].to_numpy(dtype=float)
        y_mdl = dft[model_col].to_numpy(dtype=float)

        # Courbes (markers + traits plus nets)
        h1 = ax.plot(x, y_mkt, marker="o", markersize=4, linewidth=2.0, label="Marché")[0]
        h2 = ax.plot(x, y_mdl, marker="s", markersize=4, linewidth=2.0, linestyle="--", label="Modèle")[0]

        # Style axes
        _pro_axes_style(ax)

        # Titre panel
        ax.set_title(f"{int(tenor)}Y tenor", fontsize=11, fontweight="bold", pad=8)
        ax.set_ylabel(ylabel, fontsize=10)

        # Option: petit “résidu” (Model - Market)
        if show_error_band:
            # on crée un petit axe en-dessous du subplot principal
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes

            ax_err = inset_axes(ax, width="100%", height="28%", loc="lower left",
                                bbox_to_anchor=(0, -0.38, 1, 1),
                                bbox_transform=ax.transAxes, borderpad=0)
            err = y_mdl - y_mkt
            ax_err.plot(x, err, linewidth=1.6, marker=".", markersize=4)
            ax_err.axhline(0.0, linewidth=1.0, alpha=0.5)
            ax_err.grid(True, alpha=0.15)
            ax_err.tick_params(axis="both", labelsize=8)
            ax_err.set_ylabel("Δ", fontsize=9)
            ax_err.set_xlabel("Maturité (années)", fontsize=9)

        # stocker handles pour légende globale (une seule fois)
        if handles_global is None:
            handles_global = [h1, h2]
            labels_global = ["Marché", "Modèle"]

    # masquer axes inutilisés si tenors < 4
    for j in range(n_panels, len(axes)):
        axes[j].set_axis_off()

    # Labels x sur la dernière ligne uniquement
    for ax in axes[-2:]:
        if ax.has_data():
            ax.set_xlabel("Maturité (années)", fontsize=10)

    # Titre global + sous-titre
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    if subtitle:
        fig.text(0.5, 0.955, subtitle, ha="center", va="top", fontsize=10, alpha=0.85)

    # Légende globale en haut à droite
    if handles_global is not None:
        fig.legend(handles_global, labels_global, loc="upper right", frameon=True, fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def fig_vols_by_tenor(
    df: pd.DataFrame,
    *,
    tenors=(5.0, 10.0, 20.0, 30.0),
    tenor_col="Tenor",
    x_col="Expiry",
    mkt_col="Market_Vol (Bps)",
    model_col="Model_Vol (Bps)",
    ylabel="Volatilité (Bps)",
    title="Courbe (Vol swaptions)",
    subtitle: str | None = None,
    show_error_band: bool = False,
):

    n_panels = min(len(tenors), 4)
    if n_panels <= 0:
        raise ValueError("Doit contenir au moins deux tenors.")

    fig, axes = plt.subplots(
        2, 2,
        figsize=(12.5, 8.2),
        sharex=True,
        constrained_layout=False
    )
    axes = axes.flatten()

    handles_global = None
    labels_global = None

    for idx, tenor in enumerate(tenors):
        if idx >= len(axes):
            break

        ax = axes[idx]
        dft = df[df[tenor_col] == tenor].copy()
        if dft.empty:
            ax.set_axis_off()
            continue

        dft = dft.sort_values(x_col)

        x = dft[x_col].to_numpy(dtype=float)
        y_mkt = dft[mkt_col].to_numpy(dtype=float)
        y_mdl = dft[model_col].to_numpy(dtype=float)

        h1 = ax.plot(x, y_mkt, marker="o", markersize=4, linewidth=2.0, label="Marché")[0]
        h2 = ax.plot(x, y_mdl, marker="s", markersize=4, linewidth=2.0, linestyle="--", label="Modèle")[0]

        _pro_axes_style(ax)

        ax.set_title(f"{int(tenor)}Y tenor", fontsize=11, fontweight="bold", pad=8)
        ax.set_ylabel(ylabel, fontsize=10)

        if show_error_band:
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes

            ax_err = inset_axes(ax, width="100%", height="28%", loc="lower left",
                                bbox_to_anchor=(0, -0.38, 1, 1),
                                bbox_transform=ax.transAxes, borderpad=0)
            err = y_mdl - y_mkt
            ax_err.plot(x, err, linewidth=1.6, marker=".", markersize=4)
            ax_err.axhline(0.0, linewidth=1.0, alpha=0.5)
            ax_err.grid(True, alpha=0.15)
            ax_err.tick_params(axis="both", labelsize=8)
            ax_err.set_ylabel("Δ", fontsize=9)
            ax_err.set_xlabel("Maturité (années)", fontsize=9)

        if handles_global is None:
            handles_global = [h1, h2]
            labels_global = ["Marché", "Modèle"]

    for j in range(n_panels, len(axes)):
        axes[j].set_axis_off()

    for ax in axes[-2:]:
        if ax.has_data():
            ax.set_xlabel("Maturité (années)", fontsize=10)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    if subtitle:
        fig.text(0.5, 0.955, subtitle, ha="center", va="top", fontsize=10, alpha=0.85)

    if handles_global is not None:
        fig.legend(handles_global, labels_global, loc="upper right", frameon=True, fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def fig_pfe(grid, pfe, epe=None, q: float = 0.95, title: str = "Profil PFE", subtitle: str | None = None):
    """
    Figure PFE/EPE.

    Paramètres
    ----------
    grid:
        Grille temps (années).
    pfe:
        Série PFE (même longueur que grid).
    epe:
        Série EPE (optionnelle).
    q:
        Quantile affiché dans la légende (ex 0.95).
    title / subtitle:
        Titres affichés sur la figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    grid = np.asarray(grid, dtype=float)
    pfe = np.asarray(pfe, dtype=float)
    epe = None if epe is None else np.asarray(epe, dtype=float)

    fig, ax = plt.subplots(figsize=(11, 5.5))

    # Courbe PFE + zone remplie (visuel "propre")
    ax.plot(grid, pfe, marker="o", markersize=3.5, linewidth=2.0, label=f"PFE ({int(q*100)}%)")
    ax.fill_between(grid, 0.0, pfe, alpha=0.12)

    # Courbe EPE (optionnelle)
    if epe is not None:
        ax.plot(grid, epe, marker="s", markersize=3.2, linewidth=1.8, linestyle="--", label="EPE")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    if subtitle:
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=10, alpha=0.9, va="bottom")

    ax.set_xlabel("Temps (Années)")
    ax.set_ylabel("Exposure")
    ax.yaxis.set_major_formatter(FuncFormatter(_human_money))

    ax.grid(alpha=0.25)
    ax.set_xlim(grid.min(), grid.max())
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="upper right")

    plt.tight_layout()
    return fig
