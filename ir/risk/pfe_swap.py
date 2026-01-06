# -*- coding: utf-8 -*-
"""
pfe_swap.py

Utilitaires d'exposition / PFE pour un IRS vanilla (swap) sous un simulateur de courbe.

Compatibilité
-------------
Fonctionne avec :
- HullWhiteCurveBuilder (1F) : l'objet expose zero_coupon_bond(t,T) et sim.n_paths
- HW2FCurveSim (2F)         : l'objet expose zero_coupon_bond(t,T) et n_paths

"""

from __future__ import annotations

import time
from typing import Callable, Optional, Dict, Any

import numpy as np


def _get_n_paths(curve_sim) -> int:
    """
    Déduit le nombre de scénarios Monte Carlo depuis l'objet simulateur.

    Pourquoi ?
    ----------
    Tes deux simulateurs (1F vs 2F) n’exposent pas forcément le même attribut :
    - HW2F : curve_sim.n_paths
    - HW1F : curve_sim.sim.n_paths (via HullWhiteCurveBuilder.sim)

    Retourne
    --------
    int
        Nombre de paths.

    """
    if hasattr(curve_sim, "n_paths"):
        return int(curve_sim.n_paths)
    if hasattr(curve_sim, "sim") and hasattr(curve_sim.sim, "n_paths"):
        return int(curve_sim.sim.n_paths)
    raise AttributeError(
        "Objets attendus : curve_sim.n_paths ou curve_sim.sim.n_paths."
    )


def swap_mtm_distribution_at_t(
    curve_sim,
    t: float,
    Tau,
    K: float,
    N: float,
    payer: bool = True,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    progress_ctx: Optional[Dict[str, Any]] = None,
    inner_progress: bool = False,
    inner_every: int = 3,
) -> np.ndarray:
    """
    Calcule la distribution de V(t) (MTM/PV à la date t) d'un swap vanilla.

    Idée
    ----
    On valorise le swap à la date t sur chaque scénario en utilisant les ZC bonds simulés :
      - Jambe flottante : P(t,T0) - P(t,Tm)
      - Jambe fixe      : K * A(t) où A(t)=∑ Delta_i * P(t,Ti) (annuité)

    Puis :
      V(t) = N * w * (float_leg - fixed_leg)
    où w=+1 (payer) ou -1 (receiver).

    Paramètres
    ----------
    curve_sim :
        Objet expose zero_coupon_bond(t,T) -> ndarray(n_paths,)
    t :
        Date d'évaluation (années).
    Tau :
        Échéancier des paiements [T0, T1, ..., Tm] (T0 = start/expiry côté swap).
    K :
        Taux fixe (en unité de taux, ex 0.03).
    N :
        Notionnel.
    payer :
        True => payer swap (paye fixe, reçoit float).
        False => receiver swap.
    progress_cb :
        Callback optionnel pour mise à jour UI.
    progress_ctx :
        Dict de contexte (ex {"grid_i":..., "grid_n":...}) fusionné dans le payload.
    inner_progress :
        Si True, envoie des updates pendant la boucle d'annuité (cashflows).
    inner_every :
        Fréquence des updates "cashflows" (pour éviter trop d’appels UI).

    Retourne
    --------
    np.ndarray
        V(t) sur n_paths scénarios.
    """
    # Signe selon le sens du swap
    w = 1.0 if payer else -1.0

    t = float(t)
    K = float(K)
    N = float(N)

    # On garde uniquement les dates >= t (on "raccourcit" le swap à partir de t)
    Tau_rem = [float(Ti) for Ti in Tau if float(Ti) >= t - 1e-12]
    if len(Tau_rem) < 2:
        # Si le swap est "terminé" (plus de cashflows), MTM = 0 sur tous les scénarios
        return np.zeros(_get_n_paths(curve_sim), dtype=float)

    ctx = dict(progress_ctx or {})
    ctx.update({"t": t})

    # Bornes de la jambe flottante : T0 (start) et Tm (dernier paiement)
    T0 = Tau_rem[0]
    Tm = Tau_rem[-1]

    # ZC bonds simulés pour les deux bornes de la jambe flottante
    P_t_T0 = curve_sim.zero_coupon_bond(t, T0)
    P_t_Tm = curve_sim.zero_coupon_bond(t, Tm)

    # Calcul de l'annuité A(t) = ∑ Delta_i P(t,Ti)
    annuity = 0.0
    n_cf = max(len(Tau_rem) - 1, 0)

    for i in range(1, len(Tau_rem)):
        Ti = Tau_rem[i]
        Delta = Tau_rem[i] - Tau_rem[i - 1]

        # P(t,Ti) simulé (ndarray)
        P_t_Ti = curve_sim.zero_coupon_bond(t, Ti)

        # Contribution du cashflow à l'annuité
        annuity += Delta * P_t_Ti

        if inner_progress and progress_cb is not None and n_cf > 0:
            is_last = (i == len(Tau_rem) - 1)
            if is_last or (inner_every > 0 and (i % inner_every == 0)):
                payload = {
                    **ctx,
                    "stage": "cashflows",
                    "cf_i": int(i),
                    "cf_n": int(n_cf),
                    "Ti": float(Ti),
                }
                progress_cb(payload)

    # Jambe flottante en valeur (sur chaque scénario)
    float_leg = P_t_T0 - P_t_Tm

    # Jambe fixe en valeur (sur chaque scénario)
    fixed_leg = K * annuity

    # MTM du swap sur chaque path
    V_t = N * w * (float_leg - fixed_leg)
    return V_t


def pfe_profile_swap(
    curve_sim,
    grid,
    Tau,
    K: float,
    N: float,
    payer: bool = True,
    q: float = 0.95,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    inner_progress: bool = False,
    inner_every: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcule un profil PFE_q(t) et EPE(t) sur une grille de temps.

    Définitions (sur V(t)^+ = max(V(t),0))
    --------------------------------------
    - PFE_q(t) = quantile_q(V(t)^+)
    - EPE(t)   = E[V(t)^+]

    Paramètres
    ----------
    curve_sim :
        Simulateur exposant zero_coupon_bond(t,T) -> ndarray.
    grid :
        Grille des dates (années) où on calcule l'exposition.
    Tau :
        Échéancier swap.
    K, N, payer :
        Spécifications du swap.
    q :
        Niveau de quantile (ex 0.95).
    progress_cb :
    inner_progress :
    inner_every :

    Retourne
    --------
    (pfe, epe) : tuple[np.ndarray, np.ndarray]
        Deux tableaux alignés avec grid.
    """
    # Chrono global (utile pour ETA)
    t0 = time.perf_counter()

    grid = np.asarray(grid, dtype=float)

    # Sorties
    pfe = np.zeros(len(grid), dtype=float)
    epe = np.zeros(len(grid), dtype=float)

    total_steps = int(len(grid))
    n_paths = _get_n_paths(curve_sim)

    # Boucle principale : un point de grille = un calcul de distribution V(t)
    for j, t in enumerate(grid, start=1):
        # Contexte de progression au niveau du nœud de grille
        ctx = {"grid_i": int(j), "grid_n": int(total_steps), "n_paths": int(n_paths)}

        # Distribution MTM du swap à t (ndarray n_paths)
        V_t = swap_mtm_distribution_at_t(
            curve_sim,
            float(t),
            Tau,
            K,
            N,
            payer=payer,
            progress_cb=progress_cb,
            progress_ctx=ctx,
            inner_progress=inner_progress,
            inner_every=inner_every,
        )

        # Exposition positive
        V_pos = np.maximum(V_t, 0.0)

        # PFE et EPE au temps t
        pfe_t = float(np.quantile(V_pos, q))
        epe_t = float(np.mean(V_pos))

        # Stockage
        pfe[j - 1] = pfe_t
        epe[j - 1] = epe_t

        if progress_cb is not None:
            elapsed = time.perf_counter() - t0
            pct = j / max(total_steps, 1)
            eta = (elapsed / pct - elapsed) if pct > 1e-12 else None

            progress_cb(
                {
                    "stage": "grid",
                    "grid_i": int(j),
                    "grid_n": int(total_steps),
                    "t": float(t),
                    "done": int(j),
                    "total": int(total_steps),
                    "pct": float(pct),
                    "elapsed_s": float(elapsed),
                    "eta_s": (float(eta) if eta is not None and np.isfinite(eta) else None),
                    "pfe_t": float(pfe_t),
                    "epe_t": float(epe_t),
                    "n_paths": int(n_paths),
                }
            )

    return pfe, epe
