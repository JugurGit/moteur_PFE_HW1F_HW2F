# -*- coding: utf-8 -*-
"""
pfe_plot.py

Helpers de plotting pour profils PFE/EPE (matplotlib).
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def _human_money(x, pos=None) -> str:
    """
    Formatte un nombre "monétaire" en version lisible.

    Exemples
    --------
    1200      -> "1.2k"
    1200000   -> "1.2M"
    -35000000 -> "-35.0M"

    Paramètres
    ----------
    x : float
        Valeur brute à formater (axe Y).
    pos : int | None
        Position du tick 

    Retourne
    --------
    str
        Chaîne formattée (k, M, B).
    """
    x = float(x)
    ax = abs(x)

    # On applique des suffixes en fonction de l’ordre de grandeur
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
    title: str = "Profil PFE",
    subtitle: str | None = None,
    xlabel: str = "Temps (Années)",
    ylabel: str = "Exposure",
    show_fill: bool = True,
    annotate_peak: bool = True,
    savepath: str | None = None,
):
    """
    Trace un profil PFE "propre" (et optionnellement EPE).

    Paramètres
    ----------
    grid : array-like
        Grille de temps (en années).
    pfe : array-like
        Valeurs de PFE (même longueur que grid).
    epe : array-like, optionnel
        Valeurs de EPE (même longueur que grid).
    q : float
        Quantile utilisé pour la PFE (ex: 0.95, 0.975, 0.99).
    title : str
        Titre principal du graphique.
    subtitle : str, optionnel
        Ligne d’information additionnelle (détails swap, notional, strike, paramètres modèle...).
    xlabel : str
        Label de l’axe X.
    ylabel : str
        Label de l’axe Y.
    show_fill : bool
        Si True, remplit légèrement sous la courbe PFE (effet visuel).
    annotate_peak : bool
        Si True, met en évidence et annote le maximum de la PFE.
    savepath : str, optionnel
        Si fourni, sauvegarde la figure au format PNG.

    Retourne
    --------
    (fig, ax)
        Objets Matplotlib (figure, axes) pour permettre des ajustements externes.
    """
    # Conversion en tableaux NumPy pour garantir le bon typage
    grid = np.asarray(grid, dtype=float)
    pfe = np.asarray(pfe, dtype=float)
    if epe is not None:
        epe = np.asarray(epe, dtype=float)

    # Création figure/axes (un seul panneau)
    fig, ax = plt.subplots(figsize=(11, 5.5))

    # Courbe PFE (label basé sur le quantile)
    label_pfe = f"PFE ({int(q*100)}%)"
    ax.plot(grid, pfe, marker="o", markersize=3.5, linewidth=2.0, label=label_pfe)

    # Option : remplir sous la courbe (utile visuellement)
    if show_fill:
        ax.fill_between(grid, 0.0, pfe, alpha=0.12)

    # Courbe EPE optionnelle (style différent)
    if epe is not None:
        ax.plot(
            grid,
            epe,
            marker="s",
            markersize=3.2,
            linewidth=1.8,
            linestyle="--",
            label="EPE",
        )

    # Titre + sous-titre optionnel
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    if subtitle:
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=10, alpha=0.9, va="bottom")

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.yaxis.set_major_formatter(FuncFormatter(_human_money))
    ax.grid(True, alpha=0.25)
    ax.set_xlim(grid.min(), grid.max())
    ax.set_ylim(bottom=0.0)

    # Option : annotation du pic (max PFE)
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

    # Légende + mise en page
    ax.legend(frameon=True, fontsize=10, loc="upper right")
    plt.tight_layout()

    # Option : sauvegarde en PNG
    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")

    # Affichage 
    plt.show()
    return fig, ax
