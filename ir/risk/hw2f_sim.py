# -*- coding: utf-8 -*-
"""
hw2f_sim.py

Minimal Monte Carlo wrapper for Hull–White 2F (G2++) to generate distributions of
zero-coupon bond prices P(t,T).

This file is extracted from your notebook so you can import it later without
copy/pasting cells.

Notes (important for "same results as notebook")
-----------------------------------------------
- By default, this class uses NumPy's *global* RNG (np.random.seed(seed)) exactly
  like your notebook cell. That means results are reproducible but also depend
  on other random draws in your session.
- Each call to zero_coupon_bond(t, T) draws fresh (x_t, y_t). This matches your
  notebook's behavior. (It does *not* reuse the same factors across maturities.)
"""

from __future__ import annotations
import numpy as np


class HW2FCurveSim:
    """
    Minimal simulation wrapper for HW2F to mimic curve_sim in your 1F code.

    Provides:
      - zero_coupon_bond(t, T): ndarray of P(t,T) over n_paths

    Parameters
    ----------
    curve:
        Your Curve instance (must provide discount(t)).
    model:
        HullWhite2FModel instance (must provide B_a, B_b, v2_caplet and parameters).
    n_paths:
        Number of Monte Carlo paths.
    seed:
        Random seed used when use_legacy_global_seed=True.
    use_legacy_global_seed:
        If True (default), call np.random.seed(seed) and draw via np.random.normal
        to match your notebook cell behavior.
        If False, uses a local Generator (recommended for isolation).
    """

    def __init__(
        self,
        curve,
        model,
        n_paths: int = 20000,
        seed: int = 2025,
        use_legacy_global_seed: bool = True,
    ):
        self.curve = curve
        self.model = model
        self.n_paths = int(n_paths)

        self._use_legacy = bool(use_legacy_global_seed)
        if self._use_legacy:
            np.random.seed(seed)
            self._rng = None
        else:
            self._rng = np.random.default_rng(seed)

    # -------------------------
    # Internal random utilities
    # -------------------------

    def _normal(self, size: int) -> np.ndarray:
        if self._use_legacy:
            return np.random.normal(size=size)
        return self._rng.normal(size=size)

    # -------------------------
    # Exact joint Gaussian draw
    # -------------------------

    def _simulate_xy(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Exact joint Gaussian simulation of (x_t, y_t) under Q:
          x_t = sigma ∫ e^{-a(t-s)} dW1
          y_t = eta   ∫ e^{-b(t-s)} dW2
          corr(dW1,dW2)=rho
        """
        t = float(t)

        if t <= 1e-16:
            x = np.zeros(self.n_paths)
            y = np.zeros(self.n_paths)
            return x, y

        a = float(self.model.parameters["a"])
        b = float(self.model.parameters["b"])
        rho = float(self.model.parameters["rho"])
        sigma = float(self.model.parameters["sigma"])
        eta = float(self.model.parameters["eta"])

        vx = (sigma**2 / (2.0 * a)) * (1.0 - np.exp(-2.0 * a * t))
        vy = (eta**2   / (2.0 * b)) * (1.0 - np.exp(-2.0 * b * t))
        cxy = (rho * sigma * eta / (a + b)) * (1.0 - np.exp(-(a + b) * t))

        # Sample correlated normals via stable 2x2 construction
        z1 = self._normal(self.n_paths)
        z2 = self._normal(self.n_paths)

        sx = np.sqrt(max(vx, 0.0))
        alpha = cxy / (sx + 1e-18)
        beta2 = vy - alpha**2
        beta = np.sqrt(max(beta2, 0.0))

        x = sx * z1
        y = alpha * z1 + beta * z2
        return x, y

    # -------------------------
    # ZC bond distribution
    # -------------------------

    def zero_coupon_bond(self, t: float, T: float) -> np.ndarray:
        """
        Distribution of P(t,T) over paths using affine Gaussian form:
          P(t,T) = P(0,T)/P(0,t) * exp( -B_a x_t - B_b y_t - 0.5 v^2(t,T) )

        where v^2(t,T) is taken from model.v2_caplet(expiry=t, maturity=T)
        (this matches your notebook implementation).
        """
        t = float(t)
        T = float(T)

        if T < t - 1e-12:
            raise ValueError("Need T >= t for P(t,T).")
        if abs(T - t) < 1e-12:
            return np.ones(self.n_paths)

        ratio = float(self.curve.discount(T)) / (float(self.curve.discount(t)) + 1e-18)

        x, y = self._simulate_xy(t)

        Ba = float(self.model.B_a(t, T))
        Bb = float(self.model.B_b(t, T))
        v2 = float(self.model.v2_caplet(t, T))
        adj = -0.5 * v2

        return ratio * np.exp(-Ba * x - Bb * y + adj)
