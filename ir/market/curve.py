from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
import numpy as np


class Curve:
    """
    Classe pour manipuler une courbe de marché, notamment :
    - les facteurs d’actualisation P(0,t)
    - les taux forwards instantanés f(0,t)
    avec :
    - interpolation cubique des discount factors
    - spline (lissée) sur log(P(0,t)) pour dériver un forward instantané stable
    """

    def __init__(self, time, discount_factors, smooth=1e-7):
        """
        Initialise une courbe de marché (discount curve) et construit en même temps
        les objets d’interpolation nécessaires.

        Paramètres
        ----------
        time : array_like
            Maturités (en années) des nœuds de la courbe.
        discount_factors : array_like
            Facteurs d’actualisation P(0, T) aux maturités fournies.
        smooth : float, optionnel
            Paramètre de lissage de la spline sur log(P).
        """
        self.time = np.array(time)
        self.discount_factors = np.array(discount_factors)
        self.smooth = smooth

        # Construction des interpolateurs (DF + spline de forwards instantanés)
        self._build_interpolators()

    def _build_interpolators(self):
        """
        Construit en une fois :
        1) une fonction d’interpolation des discount factors P(0,t)
        2) une spline sur ln(P(0,t)) permettant d’obtenir f(0,t) par dérivation

        Pourquoi une spline sur ln(P) ?
        -------------------------------
        En théorie :
          f(0,t) = - d/dt ln P(0,t)
        Donc si on approxime ln(P) par une spline suffisamment régulière,
        la dérivée donne un forward instantané plus stable numériquement.
        """

        self.discount_func = interp1d(
            self.time,
            self.discount_factors,
            kind="cubic",
            fill_value="extrapolate",
            bounds_error=False,
        )

        lnP = np.log(self.discount_factors)
        self.forward_spline = UnivariateSpline(self.time, lnP, s=self.smooth)

    def discount(self, t):
        """
        Retourne le facteur d’actualisation interpolé P(0,t).

        Paramètres
        ----------
        t : float ou array_like
            Temps / maturité(s) en années.

        Retourne
        --------
        float ou np.ndarray
            Valeur(s) P(0,t) interpolée(s).
        """
        return self.discount_func(t)

    def inst_forward_rate(self, t):
        """
        Retourne le taux forward instantané interpolé f(0,t).

        Formule
        -------
          f(0,t) = - d/dt ln P(0,t)

        Paramètres
        ----------
        t : float ou array_like
            Temps / maturité(s) en années.

        Retourne
        --------
        float ou np.ndarray
            Valeur(s) du forward instantané.
        """
        t = np.array(t)

        return -self.forward_spline.derivative(1)(t)

    def forward_rate(self, T1, T2):
        """
        Calcule le taux forward simple F(0; T1, T2) implicite de la courbe de DF.

        Convention (forward simple)
        ---------------------------
          F(0;T1,T2) = ( P(0,T1)/P(0,T2) - 1 ) / (T2 - T1)

        Paramètres
        ----------
        T1 : float
            Début de la période.
        T2 : float
            Fin de la période.

        Retourne
        --------
        float
            Taux forward simple entre T1 et T2.
        """
        # Récupération des DF interpolés
        P1 = self.discount(T1)
        P2 = self.discount(T2)

        # Application de la formule du forward simple (suppose T2 > T1)
        return (P1 / P2 - 1.0) / (T2 - T1)
