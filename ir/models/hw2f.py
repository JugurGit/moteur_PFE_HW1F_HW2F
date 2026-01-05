import pandas as pd
import numpy as np


class HullWhite2FModel:
    """
    Hull-White two-factor (G2++) Gaussian short-rate model - analytical building blocks.

    We use the standard G2++ factor dynamics under the risk-neutral measure:
        dx(t) = -a x(t) dt + sigma dW1(t)
        dy(t) = -b y(t) dt + eta   dW2(t)
        dW1 dW2 = rho dt

    This class focuses on *closed-form* ingredients needed for:
      - Caplet pricing via bond-option representation (needs v^2(T,S))
      - Swaption pricing via Gaussian swap-rate approximation (needs I_aa, I_bb, I_ab)

    Notes
    -----
    - The curve is assumed to provide P(0,t) and f(0,t) via the same interface as in 1F.
    - Time is in years.
    - Parameters are constants (no piecewise term structures).
    """

    def __init__(self, curve, parameters=None):
        self.curve = curve

        # Defaults (reasonable generic starting points)
        defaults = {
            "a": 0.10,
            "b": 0.02,
            "rho": -0.30,
            "sigma": 0.01,
            "eta": 0.008,
            "r0": curve.inst_forward_rate(0),
        }
        if parameters is None:
            parameters = {}

        self.parameters = {
            "a": float(parameters.get("a", defaults["a"])),
            "b": float(parameters.get("b", defaults["b"])),
            "rho": float(parameters.get("rho", defaults["rho"])),
            "sigma": float(parameters.get("sigma", defaults["sigma"])),
            "eta": float(parameters.get("eta", defaults["eta"])),
            "r0": float(parameters.get("r0", defaults["r0"])),
        }

    # --- curve interface (same spirit as HullWhiteModel 1F) --- #

    def inst_forward_rate(self, t: float) -> float:
        return self.curve.inst_forward_rate(t)

    def discount_factor(self, t: float) -> float:
        return self.curve.discount(t)

    def forward_rate(self, T1: float, T2: float) -> float:
        return self.curve.forward_rate(T1, T2)

    # --- HW2F deterministic loadings --- #

    def B_a(self, t: float, T: float) -> float:
        """ B_a(t,T) = (1 - exp(-a*(T-t)))/a """
        a = self.parameters["a"]
        return (1.0 - np.exp(-a * (T - t))) / a

    def B_b(self, t: float, T: float) -> float:
        """ B_b(t,T) = (1 - exp(-b*(T-t)))/b """
        b = self.parameters["b"]
        return (1.0 - np.exp(-b * (T - t))) / b

    # ------------------------------------------------------------------
    # Caplet ingredient: v^2(T,S) for bond option (and thus caplet)
    # ------------------------------------------------------------------

    def v2_caplet(self, T: float, S: float) -> float:
        """
        Closed-form variance v^2(T,S) entering the ZC bond option (and caplet) formula.

        With tau = S - T, and constant parameters:
            v^2(T,S) =
              (sigma^2/(2a^3)) (1 - e^{-2aT}) (1 - e^{-a tau})^2
            + (eta^2  /(2b^3)) (1 - e^{-2bT}) (1 - e^{-b tau})^2
            + (2 rho sigma eta /(a b (a+b))) (1 - e^{-(a+b)T})
                (1 - e^{-a tau})(1 - e^{-b tau})

        Parameters
        ----------
        T : float
            Option expiry / fixing time.
        S : float
            Bond maturity / payment time (S > T).

        Returns
        -------
        float
            Variance v^2(T,S) (dimensionless).
        """
        if S <= T:
            raise ValueError("v2_caplet requires S > T.")

        a = self.parameters["a"]
        b = self.parameters["b"]
        rho = self.parameters["rho"]
        sigma = self.parameters["sigma"]
        eta = self.parameters["eta"]

        tau = S - T

        # convenient exponentials
        eaT = np.exp(-2.0 * a * T)
        ebT = np.exp(-2.0 * b * T)
        eabT = np.exp(-(a + b) * T)

        ea_tau = np.exp(-a * tau)
        eb_tau = np.exp(-b * tau)

        term_a = (sigma * sigma) / (2.0 * a**3) * (1.0 - eaT) * (1.0 - ea_tau) ** 2
        term_b = (eta * eta) / (2.0 * b**3) * (1.0 - ebT) * (1.0 - eb_tau) ** 2
        term_ab = (
            2.0
            * rho
            * sigma
            * eta
            / (a * b * (a + b))
            * (1.0 - eabT)
            * (1.0 - ea_tau)
            * (1.0 - eb_tau)
        )

        return float(term_a + term_b + term_ab)

    # ------------------------------------------------------------------
    # Swaption approx ingredient: closed-form integrals I_aa, I_bb, I_ab
    # ------------------------------------------------------------------

    @staticmethod
    def I_aa(T: float, U: float, V: float, a: float) -> float:
        """
        I_aa(T;U,V) = ∫_0^T B_a(t,U) B_a(t,V) dt, with B_a(t,U)=(1-e^{-a(U-t)})/a.

        Closed form (standard case U,V >= T):
            T/a^2
          - (e^{-aU}+e^{-aV})(e^{aT}-1)/a^3
          + e^{-a(U+V)}(e^{2aT}-1)/(2a^3)
        """
        if a <= 0:
            raise ValueError("a must be > 0 in I_aa.")
        return float(
            (T / a**2)
            - ((np.exp(-a * U) + np.exp(-a * V)) * (np.exp(a * T) - 1.0) / a**3)
            + (np.exp(-a * (U + V)) * (np.exp(2.0 * a * T) - 1.0) / (2.0 * a**3))
        )

    @staticmethod
    def I_bb(T: float, U: float, V: float, b: float) -> float:
        """Same as I_aa, replacing a by b."""
        if b <= 0:
            raise ValueError("b must be > 0 in I_bb.")
        return float(
            (T / b**2)
            - ((np.exp(-b * U) + np.exp(-b * V)) * (np.exp(b * T) - 1.0) / b**3)
            + (np.exp(-b * (U + V)) * (np.exp(2.0 * b * T) - 1.0) / (2.0 * b**3))
        )

    @staticmethod
    def I_ab(T: float, U: float, V: float, a: float, b: float) -> float:
        """
        I_ab(T;U,V) = ∫_0^T B_a(t,U) B_b(t,V) dt

        Closed form (standard case U,V >= T):
            T/(ab)
          - e^{-aU}(e^{aT}-1)/(a^2 b)
          - e^{-bV}(e^{bT}-1)/(a b^2)
          + e^{-(aU+bV)}(e^{(a+b)T}-1)/(ab(a+b))
        """
        if a <= 0 or b <= 0:
            raise ValueError("a,b must be > 0 in I_ab.")
        return float(
            (T / (a * b))
            - (np.exp(-a * U) * (np.exp(a * T) - 1.0) / (a**2 * b))
            - (np.exp(-b * V) * (np.exp(b * T) - 1.0) / (a * b**2))
            + (np.exp(-(a * U + b * V)) * (np.exp((a + b) * T) - 1.0) / (a * b * (a + b)))
        )

   
   

