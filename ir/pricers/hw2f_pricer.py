import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from ir.models.hw2f import HullWhite2FModel


class HullWhite2FPricer:
    """
    Pricing engine for caplets and swaptions under Hull-White 2F (G2++):

    - Caplet pricing: via ZC bond put using closed-form bond option variance v^2(T,S)
    - Swaption pricing: via Gaussian swap-rate approximation and Bachelier pricing

    Notes
    -----
    - Single-curve setup: discounting and forwarding from the same curve.
    - Accruals are approximated as Delta = Tau[i] - Tau[i-1] (same convention as your 1F pricer).
    """

    def __init__(self, curve, hw2f_params=None):
        self.curve = curve
        self.model = HullWhite2FModel(curve, hw2f_params)

    # -------------------------
    # Helpers (curve quantities)
    # -------------------------

    def discount_factor(self, t: float) -> float:
        return self.model.discount_factor(t)

    def _annuity_and_swap_rate_0(self, Tau):
        """
        Compute A0 and S0 for a fixed-leg schedule Tau = [T0, T1, ..., Tn].

        A0 = sum_i Delta_i P(0,Ti)
        S0 = (P(0,T0) - P(0,Tn)) / A0
        """
        if len(Tau) < 2:
            raise ValueError("Tau must contain at least [T0, Tn].")

        T0 = float(Tau[0])
        Tn = float(Tau[-1])

        A0 = 0.0
        for i in range(1, len(Tau)):
            Ti = float(Tau[i])
            delta = float(Tau[i] - Tau[i - 1])
            A0 += delta * self.discount_factor(Ti)

        if A0 <= 0:
            raise ValueError("Annuity A0 must be > 0.")

        S0 = (self.discount_factor(T0) - self.discount_factor(Tn)) / A0
        return float(A0), float(S0)

    def _swaption_weights_frozen(self, Tau):
        """
        Build frozen weights c_j for dates U_j = Tau[j] in the Gaussian swap-rate approx.

        We approximate:
          dS(t) ≈ sum_j c_j * (dP(t, U_j) / P(t, U_j))  (up to deterministic loadings)

        The frozen weights are:
          c(T0) += P(0,T0)/A0
          c(Tn) += -P(0,Tn)/A0
          c(Ti) += -(S0/A0) * Delta_i * P(0,Ti) for i=1..n
        (So the last date Tn has two contributions: numerator + annuity.)
        """
        A0, S0 = self._annuity_and_swap_rate_0(Tau)

        U = [float(x) for x in Tau]
        m = len(U)
        c = np.zeros(m, dtype=float)

        # Numerator contributions: +P(0,T0)/A0 and -P(0,Tn)/A0
        c[0] += self.discount_factor(U[0]) / A0
        c[-1] += -self.discount_factor(U[-1]) / A0

        # Annuity contribution: -(S0/A0) * Delta_i * P(0,Ti)
        for i in range(1, m):
            Ti = U[i]
            delta = U[i] - U[i - 1]
            c[i] += -(S0 / A0) * delta * self.discount_factor(Ti)

        return U, c, A0, S0

    # ---------------------------------------
    # HW2F bond options (closed form via v^2)
    # ---------------------------------------

    def zero_bond_put_hw2f(self, T: float, S: float, K: float) -> float:
        """
        Put option on ZC bond P(T,S) with strike K (bond price strike), priced at time 0.

        Put = K P(0,T) N(-d2) - P(0,S) N(-d1)
        d1 = [ln(P(0,S)/(K P(0,T))) + 0.5 v^2]/v
        d2 = d1 - v
        where v^2 = model.v2_caplet(T,S).
        """
        T = float(T)
        S = float(S)
        K = float(K)
        if S <= T:
            raise ValueError("Bond option requires S > T.")
        if K <= 0:
            raise ValueError("Bond option strike K must be > 0.")

        P0T = self.discount_factor(T)
        P0S = self.discount_factor(S)

        v2 = self.model.v2_caplet(T, S)
        v = np.sqrt(max(v2, 0.0))

        if v < 1e-16:
            # Nearly deterministic
            return float(max(K * P0T - P0S, 0.0))

        ln_term = np.log(P0S / (K * P0T))
        d1 = (ln_term + 0.5 * v2) / v
        d2 = d1 - v

        put = K * P0T * norm.cdf(-d2) - P0S * norm.cdf(-d1)
        return float(put)

    def zero_bond_call_hw2f(self, T: float, S: float, K: float) -> float:
        """
        Call option on ZC bond P(T,S), priced at time 0.

        Call = P(0,S) N(d1) - K P(0,T) N(d2)
        """
        T = float(T)
        S = float(S)
        K = float(K)
        if S <= T:
            raise ValueError("Bond option requires S > T.")
        if K <= 0:
            raise ValueError("Bond option strike K must be > 0.")

        P0T = self.discount_factor(T)
        P0S = self.discount_factor(S)

        v2 = self.model.v2_caplet(T, S)
        v = np.sqrt(max(v2, 0.0))

        if v < 1e-16:
            return float(max(P0S - K * P0T, 0.0))

        ln_term = np.log(P0S / (K * P0T))
        d1 = (ln_term + 0.5 * v2) / v
        d2 = d1 - v

        call = P0S * norm.cdf(d1) - K * P0T * norm.cdf(d2)
        return float(call)

    # -----------------------
    # Caplet pricing under 2F
    # -----------------------

    def caplet_hw2f(self, T1: float, T2: float, N: float, K: float) -> float:
        """
        Caplet PV under HW2F via ZC bond put:
            Caplet = N * (1 + K*Delta) * PutZC(T1, T2, 1/(1+K*Delta))
        with Delta = T2 - T1.
        """
        T1 = float(T1)
        T2 = float(T2)
        N = float(N)
        K = float(K)

        if T2 <= T1:
            raise ValueError("Caplet requires T2 > T1.")
        if N <= 0:
            raise ValueError("Notional N must be > 0.")
        if K < 0:
            raise ValueError("Strike K must be >= 0.")

        Delta = T2 - T1
        K_bond = 1.0 + K * Delta
        if K_bond <= 0:
            raise ValueError("Invalid (1 + K*Delta) <= 0.")

        # Bond option strike on P(T1, T2)
        K_zc = 1.0 / K_bond

        put_zc = self.zero_bond_put_hw2f(T1, T2, K_zc)
        caplet = N * K_bond * put_zc
        return float(caplet)

    # ----------------------------------------------
    # Swaption pricing under 2F via Gaussian approx
    # ----------------------------------------------

    def swaption_approx_hw2f(self, Tau, N: float, K: float, payer: bool = True) -> float:
        """
        Swaption PV using:
          - frozen weights Gaussian swap-rate approximation
          - HW2F closed-form integrals I_aa, I_bb, I_ab
          - Bachelier (normal) pricing on swap rate

        Tau : list[float]
            [T0, T1, ..., Tn] fixed payment schedule (T0 is option expiry / swap start)
        """
        Tau = [float(x) for x in Tau]
        N = float(N)
        K = float(K)
        if len(Tau) < 2:
            raise ValueError("Tau must contain at least [T0, Tn].")
        if N <= 0:
            raise ValueError("Notional N must be > 0.")

        T = Tau[0]  # expiry
        if T <= 0:
            raise ValueError("Swaption expiry T must be > 0 for this method.")

        # Build frozen weights and swap quantities at time 0
        U, c, A0, S0 = self._swaption_weights_frozen(Tau)

        # Model parameters
        a = self.model.parameters["a"]
        b = self.model.parameters["b"]
        rho = self.model.parameters["rho"]
        sigma = self.model.parameters["sigma"]
        eta = self.model.parameters["eta"]

        # Compute Qaa, Qbb, Qab via double sums (small matrices => fine)
        Qaa = 0.0
        Qbb = 0.0
        Qab = 0.0

        for i, Ui in enumerate(U):
            ci = c[i]
            if ci == 0.0:
                continue
            for j, Uj in enumerate(U):
                cj = c[j]
                if cj == 0.0:
                    continue
                Qaa += ci * cj * HullWhite2FModel.I_aa(T, Ui, Uj, a)
                Qbb += ci * cj * HullWhite2FModel.I_bb(T, Ui, Uj, b)
                Qab += ci * cj * HullWhite2FModel.I_ab(T, Ui, Uj, a, b)

        varS = (sigma * sigma) * Qaa + (eta * eta) * Qbb + 2.0 * rho * sigma * eta * Qab
        varS = float(max(varS, 0.0))

        # Convert to normal vol on swap rate
        if varS < 1e-30:
            w = 1.0 if payer else -1.0
            return float(N * A0 * max(w * (S0 - K), 0.0))

        sigmaN = np.sqrt(varS / T)

        # Bachelier pricing
        d = (S0 - K) / (sigmaN * np.sqrt(T))
        w = 1.0 if payer else -1.0

        price = N * A0 * (w * (S0 - K) * norm.cdf(w * d) + sigmaN * np.sqrt(T) * norm.pdf(d))
        return float(price)
