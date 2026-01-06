import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from ir.models.hw2f import HullWhite2FModel


class HullWhite2FPricer:
    """
    Moteur de pricing pour caplets et swaptions sous Hull–White 2 facteurs (G2++).

    Fonctionnalités
    ---------------
    - Caplet : via put sur ZC bond avec variance fermée v^2(T,S) (formule d'option sur bond).
    - Swaption : via approximation gaussienne du swap-rate (poids figés) + pricing Bachelier (normal).

    Notes
    -----
    - Hypothèse single-curve : discounting et forwarding via la même courbe.
    - Accruals : Delta_i ≈ Tau[i] - Tau[i-1] (même convention que ton pricer 1F).
    """

    def __init__(self, curve, hw2f_params=None):
        """
        Initialise le pricer 2F.

        Paramètres
        ----------
        curve : Curve
            Courbe d’actualisation P(0,t).
        hw2f_params : dict | None
            Paramètres G2++ : {'a','b','rho','sigma','eta','r0'}.
        """
        self.curve = curve
        self.model = HullWhite2FModel(curve, hw2f_params)

    # -------------------------
    # Helpers (quantités courbe)
    # -------------------------

    def discount_factor(self, t: float) -> float:
        """
        Raccourci : renvoie P(0,t) via le modèle/courbe.

        Paramètre
        ---------
        t : float
            Temps en années.

        Retourne
        --------
        float
            Discount factor P(0,t).
        """
        return self.model.discount_factor(t)

    def _annuity_and_swap_rate_0(self, Tau):
        """
        Calcule l'annuité A0 et le swap rate forward S0 à t=0 pour un schedule fixe.

        Pour Tau = [T0, T1, ..., Tn] :
          - Delta_i = Tau[i] - Tau[i-1]
          - A0 = sum_{i=1..n} Delta_i * P(0,Ti)
          - S0 = (P(0,T0) - P(0,Tn)) / A0

        Paramètres
        ----------
        Tau : list[float]
            Dates de paiement de la jambe fixe, avec T0 = start/expiry et Tn = dernière date.

        Retourne
        --------
        (float, float)
            (A0, S0)

        """
        if len(Tau) < 2:
            raise ValueError("Tau doit contenir au moins [T0, Tn].")

        T0 = float(Tau[0])
        Tn = float(Tau[-1])

        # Annuité A0
        A0 = 0.0
        for i in range(1, len(Tau)):
            Ti = float(Tau[i])
            delta = float(Tau[i] - Tau[i - 1])
            A0 += delta * self.discount_factor(Ti)

        if A0 <= 0:
            raise ValueError("Annuity A0 doit être > 0.")

        # Swap rate forward S0
        S0 = (self.discount_factor(T0) - self.discount_factor(Tn)) / A0
        return float(A0), float(S0)

    def _swaption_weights_frozen(self, Tau):
        """
        Construit les poids figés (frozen weights) c_j pour l'approximation gaussienne du swap-rate.

        Idée
        ----
        On approxime le swap rate comme combinaison linéaire des log-bonds :
          dS(t) ≈ sum_j c_j * (dP(t,U_j)/P(t,U_j))   (à des loadings déterministes près)

        Poids figés (à t=0), avec U_j = Tau[j] :
          - c(T0) +=  P(0,T0) / A0
          - c(Tn) += -P(0,Tn) / A0
          - c(Ti) += -(S0/A0) * Delta_i * P(0,Ti)   pour i=1..n
        (Donc Tn reçoit potentiellement deux contributions : numerator + annuity.)

        Paramètres
        ----------
        Tau : list[float]
            Schedule fixe [T0, T1, ..., Tn] où T0 = expiry/start.

        Retourne
        --------
        (U, c, A0, S0)
            U : list[float] les dates
            c : np.ndarray les poids figés 
            A0 : float annuité
            S0 : float swap rate forward
        """
        A0, S0 = self._annuity_and_swap_rate_0(Tau)

        U = [float(x) for x in Tau]
        m = len(U)
        c = np.zeros(m, dtype=float)

        # Contributions du numérateur (swap parity)
        c[0] += self.discount_factor(U[0]) / A0
        c[-1] += -self.discount_factor(U[-1]) / A0

        # Contribution annuité : -(S0/A0) * Delta_i * P(0,Ti)
        for i in range(1, m):
            Ti = U[i]
            delta = U[i] - U[i - 1]
            c[i] += -(S0 / A0) * delta * self.discount_factor(Ti)

        return U, c, A0, S0

    # ---------------------------------------
    # Options sur ZC bond (fermé via v^2(T,S))
    # ---------------------------------------

    def zero_bond_put_hw2f(self, T: float, S: float, K: float) -> float:
        """
        Put européen sur ZC bond P(T,S) de strike K (strike sur le prix du bond),
        valorisé à t=0 sous HW2F.

        Formule (analogue "Black" sur ZC bond) :
          Put = K P(0,T) N(-d2) - P(0,S) N(-d1)
          d1 = [ln(P(0,S)/(K P(0,T))) + 0.5 v^2] / v
          d2 = d1 - v
        où v^2 = model.v2_caplet(T,S).

        Paramètres
        ----------
        T : float
            Expiry de l'option (en années).
        S : float
            Maturité du bond (S > T).
        K : float
            Strike (prix du bond) > 0.

        Retourne
        --------
        float
            PV du put.
        """
        T = float(T)
        S = float(S)
        K = float(K)

        if S <= T:
            raise ValueError("Bond option doit avoir S > T.")
        if K <= 0:
            raise ValueError("Bond option strike K doit être > 0.")

        P0T = self.discount_factor(T)
        P0S = self.discount_factor(S)

        # Variance v^2(T,S) (fermée) puis vol v
        v2 = self.model.v2_caplet(T, S)
        v = np.sqrt(max(v2, 0.0))

        if v < 1e-16:
            # Cas quasi déterministe : payoff intrinsèque actualisé
            return float(max(K * P0T - P0S, 0.0))

        ln_term = np.log(P0S / (K * P0T))
        d1 = (ln_term + 0.5 * v2) / v
        d2 = d1 - v

        put = K * P0T * norm.cdf(-d2) - P0S * norm.cdf(-d1)
        return float(put)

    def zero_bond_call_hw2f(self, T: float, S: float, K: float) -> float:
        """
        Call européen sur ZC bond P(T,S) de strike K, valorisé à t=0 sous HW2F.

        Formule :
          Call = P(0,S) N(d1) - K P(0,T) N(d2)
        avec d1, d2 définis comme dans zero_bond_put_hw2f.

        Paramètres
        ----------
        T : float
            Expiry.
        S : float
            Maturité (S > T).
        K : float
            Strike > 0.

        Retourne
        --------
        float
            PV du call.
        """
        T = float(T)
        S = float(S)
        K = float(K)

        if S <= T:
            raise ValueError("Bond option doit avoir S > T.")
        if K <= 0:
            raise ValueError("Bond option strike K doit être > 0.")

        P0T = self.discount_factor(T)
        P0S = self.discount_factor(S)

        v2 = self.model.v2_caplet(T, S)
        v = np.sqrt(max(v2, 0.0))

        if v < 1e-16:
            # Cas quasi déterministe : payoff intrinsèque actualisé
            return float(max(P0S - K * P0T, 0.0))

        ln_term = np.log(P0S / (K * P0T))
        d1 = (ln_term + 0.5 * v2) / v
        d2 = d1 - v

        call = P0S * norm.cdf(d1) - K * P0T * norm.cdf(d2)
        return float(call)

    # -----------------------
    # Caplet pricing sous 2F
    # -----------------------

    def caplet_hw2f(self, T1: float, T2: float, N: float, K: float) -> float:
        """
        PV d'un caplet sous HW2F via put sur ZC bond.

        Relation standard :
          Caplet = N * (1 + K*Delta) * PutZC(T1, T2, 1/(1+K*Delta))
        avec Delta = T2 - T1.

        Paramètres
        ----------
        T1 : float
            Fixing (expiry du caplet).
        T2 : float
            Paiement (T2 > T1).
        N : float
            Notional.
        K : float
            Strike en "rate units" (ex: 0.03).

        Retourne
        --------
        float
            PV du caplet.
        """
        T1 = float(T1)
        T2 = float(T2)
        N = float(N)
        K = float(K)

        if T2 <= T1:
            raise ValueError("Caplet doit avoir T2 > T1.")
        if N <= 0:
            raise ValueError("Notional N doit être > 0.")
        if K < 0:
            raise ValueError("Strike K doit être >= 0.")

        Delta = T2 - T1
        K_bond = 1.0 + K * Delta
        if K_bond <= 0:
            raise ValueError("(1 + K*Delta) <= 0 n'est pas valide.")

        # Strike de l'option sur P(T1,T2)
        K_zc = 1.0 / K_bond

        put_zc = self.zero_bond_put_hw2f(T1, T2, K_zc)
        caplet = N * K_bond * put_zc
        return float(caplet)

    # ----------------------------------------------
    # Swaption pricing sous 2F (approx gaussienne)
    # ----------------------------------------------

    def swaption_approx_hw2f(self, Tau, N: float, K: float, payer: bool = True) -> float:
        """
        PV d'une swaption via :
          - approximation gaussienne du swap rate avec poids figés
          - intégrales fermées HW2F : I_aa, I_bb, I_ab
          - pricing Bachelier (normal) sur le swap rate

        Paramètres
        ----------
        Tau : list[float]
            Schedule fixe [T0, T1, ..., Tn] (T0 = expiry / start).
        N : float
            Notional.
        K : float
            Strike en "rate units" (ex: 0.03).
        payer : bool
            True = payer swaption ; False = receiver swaption.

        Retourne
        --------
        float
            PV de la swaption.

        Remarques
        ---------
        - Ici, on utilise S0 (swap rate forward) et A0 (annuité) calculés à t=0.
        - La variance du swap rate à l'expiry T est :
            Var[S(T)] = sigma^2 Qaa + eta^2 Qbb + 2 rho sigma eta Qab
          où Q.. proviennent des doubles sommes sur les poids figés et les intégrales I_.. .
        """
        Tau = [float(x) for x in Tau]
        N = float(N)
        K = float(K)

        # Vérifications d'entrée
        if len(Tau) < 2:
            raise ValueError("Tau doit contenir au moins [T0, Tn].")
        if N <= 0:
            raise ValueError("Notional N doit être > 0.")

        T = Tau[0]  # expiry
        if T <= 0:
            # Cas expiry immédiate non géré par cette approximation
            raise ValueError("Swaption expiry T doit être > 0 pour cette méthode.")

        # Construction des poids figés + A0, S0
        U, c, A0, S0 = self._swaption_weights_frozen(Tau)

        # Paramètres du modèle
        a = self.model.parameters["a"]
        b = self.model.parameters["b"]
        rho = self.model.parameters["rho"]
        sigma = self.model.parameters["sigma"]
        eta = self.model.parameters["eta"]

        # Calcul des Qaa, Qbb, Qab par double somme
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

        # Variance du swap rate à l'expiry
        varS = (sigma * sigma) * Qaa + (eta * eta) * Qbb + 2.0 * rho * sigma * eta * Qab
        varS = float(max(varS, 0.0))

        # Cas variance quasi nulle : payoff intrinsèque à l'expiry (actualisé via annuité)
        if varS < 1e-30:
            w = 1.0 if payer else -1.0
            return float(N * A0 * max(w * (S0 - K), 0.0))

        # Conversion variance -> vol normal du swap rate (Bachelier) :
        # Var[S(T)] = sigmaN^2 * T  => sigmaN = sqrt(varS / T)
        sigmaN = np.sqrt(varS / T)

        # Pricing Bachelier
        d = (S0 - K) / (sigmaN * np.sqrt(T))
        w = 1.0 if payer else -1.0

        price = N * A0 * (w * (S0 - K) * norm.cdf(w * d) + sigmaN * np.sqrt(T) * norm.pdf(d))
        return float(price)
