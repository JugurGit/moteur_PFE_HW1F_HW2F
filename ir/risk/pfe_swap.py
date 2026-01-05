# -*- coding: utf-8 -*-
"""
pfe_swap.py

Exposure / PFE utilities for a vanilla IRS (swap) under a curve simulator.

Compatibility
-------------
Works with:
- HullWhiteCurveBuilder (1F): object has zero_coupon_bond(t,T) and sim.n_paths
- HW2FCurveSim (2F): object has zero_coupon_bond(t,T) and n_paths

Patch (progress)
----------------
- Adds optional progress callback to display live progress in Streamlit:
  * per grid node (always)
  * optional intra-node progress (per cashflow) for long Tau schedules
"""

from __future__ import annotations

import time
from typing import Callable, Optional, Dict, Any

import numpy as np


def _get_n_paths(curve_sim) -> int:
    """
    Try to infer the number of Monte Carlo paths from the simulator object.
    """
    if hasattr(curve_sim, "n_paths"):
        return int(curve_sim.n_paths)
    if hasattr(curve_sim, "sim") and hasattr(curve_sim.sim, "n_paths"):
        return int(curve_sim.sim.n_paths)
    raise AttributeError(
        "Cannot infer number of paths. Expected curve_sim.n_paths or curve_sim.sim.n_paths."
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
    Distribution de V(t) (PV à t) d'un swap vanilla.

    Parameters
    ----------
    curve_sim:
        Any object exposing zero_coupon_bond(t,T) -> ndarray.
    t:
        Evaluation time in years.
    Tau:
        Payment schedule [T0, T1, ..., Tm].
    K:
        Fixed rate (in rate units, e.g. 0.03 for 3%).
    N:
        Notional.
    payer:
        True for payer swap, False for receiver swap.
    progress_cb:
        Optional callback called during computation (to update UI).
    progress_ctx:
        Context dict (e.g. {"grid_i":..., "grid_n":...}) merged into callback payload.
    inner_progress:
        If True, calls progress_cb during annuity loop (cashflow-by-cashflow).
    inner_every:
        Call progress_cb every `inner_every` cashflows (to avoid too frequent UI updates).

    Returns
    -------
    ndarray shape (n_paths,)
        Distribution of V(t) across paths.
    """
    w = 1.0 if payer else -1.0
    t = float(t)
    K = float(K)
    N = float(N)

    Tau_rem = [float(Ti) for Ti in Tau if float(Ti) >= t - 1e-12]
    if len(Tau_rem) < 2:
        return np.zeros(_get_n_paths(curve_sim), dtype=float)

    ctx = dict(progress_ctx or {})
    ctx.update({"t": t})

    T0 = Tau_rem[0]
    Tm = Tau_rem[-1]

    # ZC for float leg endpoints
    P_t_T0 = curve_sim.zero_coupon_bond(t, T0)
    P_t_Tm = curve_sim.zero_coupon_bond(t, Tm)

    # fixed leg annuity
    annuity = 0.0
    n_cf = max(len(Tau_rem) - 1, 0)

    for i in range(1, len(Tau_rem)):
        Ti = Tau_rem[i]
        Delta = Tau_rem[i] - Tau_rem[i - 1]
        P_t_Ti = curve_sim.zero_coupon_bond(t, Ti)
        annuity += Delta * P_t_Ti

        if inner_progress and progress_cb is not None and n_cf > 0:
            # throttle UI updates
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

    float_leg = P_t_T0 - P_t_Tm
    fixed_leg = K * annuity

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
    Retourne PFE_q(t) et EPE(t) sur une grille.

    Parameters
    ----------
    curve_sim:
        Simulator with zero_coupon_bond(t,T) -> ndarray.
    grid:
        Times where to compute exposures.
    Tau:
        Swap payment schedule.
    K, N, payer:
        Swap specs.
    q:
        Quantile level for PFE (e.g. 0.95).
    progress_cb:
        Optional callback called during the run to update UI with progress.
        Payload keys (typical):
          - stage: "grid" or "cashflows"
          - grid_i, grid_n, t
          - done, total, pct
          - elapsed_s, eta_s
          - pfe_t, epe_t (for stage="grid")
    inner_progress:
        If True, emits extra progress during the cashflow loop inside each grid node.
    inner_every:
        Throttle for inner progress updates.

    Returns
    -------
    (pfe, epe): tuple of ndarrays
        Arrays aligned with grid.
    """
    t0 = time.perf_counter()
    grid = np.asarray(grid, dtype=float)

    pfe = np.zeros(len(grid), dtype=float)
    epe = np.zeros(len(grid), dtype=float)

    total_steps = int(len(grid))
    n_paths = _get_n_paths(curve_sim)

    for j, t in enumerate(grid, start=1):
        ctx = {"grid_i": int(j), "grid_n": int(total_steps), "n_paths": int(n_paths)}

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

        V_pos = np.maximum(V_t, 0.0)
        pfe_t = float(np.quantile(V_pos, q))
        epe_t = float(np.mean(V_pos))

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
