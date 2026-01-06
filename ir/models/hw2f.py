import pandas as pd
import numpy as np


class HullWhite2FModel:
    """
    Modèle Hull–White 2 facteurs (G2++) gaussien — briques analytiques.

    On utilise la dynamique standard des facteurs G2++ sous la mesure risque-neutre :
        dx(t) = -a x(t) dt + sigma dW1(t)
        dy(t) = -b y(t) dt + eta   dW2(t)
        dW1 dW2 = rho dt

    Cette classe se concentre sur les ingrédients *fermés* nécessaires pour :
      - le pricing de caplets via représentation en option sur ZC bond (besoin de v^2(T,S))
      - le pricing de swaptions via approximation gaussienne du swap rate (besoin de I_aa, I_bb, I_ab)

    Notes
    -----
    - La courbe doit fournir P(0,t) et f(0,t) via la même interface que le 1F.
    - Le temps est exprimé en années.
    - Les paramètres sont constants.
    """

    def __init__(self, curve, parameters=None):
        """
        Initialise le modèle HW2F à partir d'une courbe de marché et de paramètres.

        Paramètres
        ----------
        curve : Curve
            Courbe d'actualisation (discount curve) avec discount(t) et inst_forward_rate(t).
        parameters : dict, optionnel
            Paramètres du modèle :
              - a, b : vitesses de mean reversion
              - rho  : corrélation entre les Brownien dW1 et dW2
              - sigma, eta : volatilités des facteurs x et y
              - r0 : taux court initial (souvent f(0,0))
        """
        self.curve = curve

        # Valeurs par défaut (points de départ génériques pour calibration)
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

    # --- interface "courbe" (même esprit que HullWhiteModel 1F) --- #

    def inst_forward_rate(self, t: float) -> float:
        """
        Renvoie le forward instantané f(0,t) via la courbe.

        Paramètre
        ---------
        t : float
            Temps en années.

        Retourne
        --------
        float
            f(0,t).
        """
        return self.curve.inst_forward_rate(t)

    def discount_factor(self, t: float) -> float:
        """
        Renvoie le discount factor P(0,t) via la courbe.

        Paramètre
        ---------
        t : float
            Temps en années.

        Retourne
        --------
        float
            P(0,t).
        """
        return self.curve.discount(t)

    def forward_rate(self, T1: float, T2: float) -> float:
        """
        Renvoie le forward simple implicite entre T1 et T2 via la courbe.

        Paramètres
        ----------
        T1, T2 : float
            Bornes de la période.

        Retourne
        --------
        float
            F(0;T1,T2).
        """
        return self.curve.forward_rate(T1, T2)

    # --- chargements déterministes (loadings) HW2F --- #

    def B_a(self, t: float, T: float) -> float:
        """
        Fonction de chargement B_a(t,T) associée au facteur x(t).

        Formule :
            B_a(t,T) = (1 - exp(-a*(T-t))) / a

        Paramètres
        ----------
        t : float
            Temps courant.
        T : float
            Maturité.

        Retourne
        --------
        float
            B_a(t,T).
        """
        a = self.parameters["a"]
        return (1.0 - np.exp(-a * (T - t))) / a

    def B_b(self, t: float, T: float) -> float:
        """
        Fonction de chargement B_b(t,T) associée au facteur y(t).

        Formule :
            B_b(t,T) = (1 - exp(-b*(T-t))) / b
        """
        b = self.parameters["b"]
        return (1.0 - np.exp(-b * (T - t))) / b

    # ------------------------------------------------------------------
    # Ingrédient caplet : v^2(T,S) pour option sur ZC bond (donc caplet)
    # ------------------------------------------------------------------

    def v2_caplet(self, T: float, S: float) -> float:
        """
        Variance fermée v^2(T,S) intervenant dans la formule d'option sur ZC bond
        (et donc dans le pricing d'un caplet via la représentation standard).

        Avec tau = S - T, et paramètres constants :
            v^2(T,S) =
              (sigma^2/(2a^3)) (1 - e^{-2aT}) (1 - e^{-a tau})^2
            + (eta^2  /(2b^3)) (1 - e^{-2bT}) (1 - e^{-b tau})^2
            + (2 rho sigma eta /(a b (a+b))) (1 - e^{-(a+b)T})
                (1 - e^{-a tau})(1 - e^{-b tau})

        Paramètres
        ----------
        T : float
            Date d'expiry / fixing (T > 0).
        S : float
            Date de paiement / maturité du bond (S > T).

        Retourne
        --------
        float
            Variance v^2(T,S) (sans dimension).
        """
        if S <= T:
            raise ValueError("v2_caplet doit avoir S > T.")

        a = self.parameters["a"]
        b = self.parameters["b"]
        rho = self.parameters["rho"]
        sigma = self.parameters["sigma"]
        eta = self.parameters["eta"]

        tau = S - T

        eaT = np.exp(-2.0 * a * T)
        ebT = np.exp(-2.0 * b * T)
        eabT = np.exp(-(a + b) * T)

        ea_tau = np.exp(-a * tau)
        eb_tau = np.exp(-b * tau)

        # Contribution du facteur x (paramètres a, sigma)
        term_a = (sigma * sigma) / (2.0 * a**3) * (1.0 - eaT) * (1.0 - ea_tau) ** 2

        # Contribution du facteur y (paramètres b, eta)
        term_b = (eta * eta) / (2.0 * b**3) * (1.0 - ebT) * (1.0 - eb_tau) ** 2

        # Terme croisé (corrélation rho entre les deux facteurs)
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
    # Ingrédients swaption (approx swap-rate) : intégrales I_aa, I_bb, I_ab
    # ------------------------------------------------------------------

    @staticmethod
    def I_aa(T: float, U: float, V: float, a: float) -> float:
        """
        Calcule l'intégrale fermée :
            I_aa(T;U,V) = ∫_0^T B_a(t,U) B_a(t,V) dt
        avec :
            B_a(t,U) = (1 - exp(-a*(U - t))) / a

        Forme fermée (cas standard U,V >= T) :
            T/a^2
          - (e^{-aU}+e^{-aV})(e^{aT}-1)/a^3
          + e^{-a(U+V)}(e^{2aT}-1)/(2a^3)

        Paramètres
        ----------
        T : float
            Horizon d'intégration (expiry swaption, typiquement).
        U, V : float
            Dates de flux (ou points de grille) intervenant dans l'approx du swap rate.
        a : float
            Mean reversion du facteur x.

        Retourne
        --------
        float
            Valeur de l'intégrale I_aa.

        """
        if a <= 0:
            raise ValueError("a doit être > 0 dans I_aa.")

        return float(
            (T / a**2)
            - ((np.exp(-a * U) + np.exp(-a * V)) * (np.exp(a * T) - 1.0) / a**3)
            + (np.exp(-a * (U + V)) * (np.exp(2.0 * a * T) - 1.0) / (2.0 * a**3))
        )

    @staticmethod
    def I_bb(T: float, U: float, V: float, b: float) -> float:
        """
        Même intégrale que I_aa, mais pour le facteur y (paramètre b).

        Paramètres
        ----------
        T : float
            Horizon d'intégration.
        U, V : float
            Dates / points de grille.
        b : float
            Mean reversion du facteur y.

        Retourne
        --------
        float
            Valeur de I_bb.

        """
        if b <= 0:
            raise ValueError("b doit être > 0 dans I_bb.")

        return float(
            (T / b**2)
            - ((np.exp(-b * U) + np.exp(-b * V)) * (np.exp(b * T) - 1.0) / b**3)
            + (np.exp(-b * (U + V)) * (np.exp(2.0 * b * T) - 1.0) / (2.0 * b**3))
        )

    @staticmethod
    def I_ab(T: float, U: float, V: float, a: float, b: float) -> float:
        """
        Calcule l'intégrale croisée :
            I_ab(T;U,V) = ∫_0^T B_a(t,U) B_b(t,V) dt

        Formule fermée (cas standard U,V >= T) :
            T/(ab)
          - e^{-aU}(e^{aT}-1)/(a^2 b)
          - e^{-bV}(e^{bT}-1)/(a b^2)
          + e^{-(aU+bV)}(e^{(a+b)T}-1)/(ab(a+b))

        Paramètres
        ----------
        T : float
            Horizon d'intégration.
        U : float
            Date/point associé à B_a.
        V : float
            Date/point associé à B_b.
        a : float
            Mean reversion facteur x.
        b : float
            Mean reversion facteur y.

        Retourne
        --------
        float
            Valeur de I_ab.

        """
        if a <= 0 or b <= 0:
            raise ValueError("a,b doit être > 0 dans I_ab.")

        return float(
            (T / (a * b))
            - (np.exp(-a * U) * (np.exp(a * T) - 1.0) / (a**2 * b))
            - (np.exp(-b * V) * (np.exp(b * T) - 1.0) / (a * b**2))
            + (
                np.exp(-(a * U + b * V))
                * (np.exp((a + b) * T) - 1.0)
                / (a * b * (a + b))
            )
        )
