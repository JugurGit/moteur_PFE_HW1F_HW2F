# -*- coding: utf-8 -*-
"""
hw2f_sim.py

Wrapper Monte Carlo pour Hull–White 2 facteurs (G2++, aka HW2F),
afin de générer des distributions de prix de ZC bonds P(t,T).

"""

from __future__ import annotations
import numpy as np


class HW2FCurveSim:
    """
    Wrapper de simulation pour HW2F.

    Fournit
    --------
    - zero_coupon_bond(t, T) : ndarray des P(t,T) sur n_paths scénarios

    Paramètres
    ----------
    curve :
        Instance de Curve (doit fournir discount(t)).
    model :
        Instance de HullWhite2FModel (doit fournir B_a, B_b, v2_caplet et parameters).
    n_paths :
        Nombre de scénarios Monte Carlo.
    seed :
    """

    def __init__(
        self,
        curve,
        model,
        n_paths: int = 20000,
        seed: int = 2025,
        use_legacy_global_seed: bool = True,
    ):
        # Stockage des objets "marché" et "modèle"
        self.curve = curve
        self.model = model

        # Paramètre Monte Carlo
        self.n_paths = int(n_paths)

        self._use_legacy = bool(use_legacy_global_seed)
        if self._use_legacy:
            np.random.seed(seed)
            self._rng = None
        else:
            self._rng = np.random.default_rng(seed)

    # -------------------------
    # Utilitaires RNG internes
    # -------------------------

    def _normal(self, size: int) -> np.ndarray:
        """
        Tire des normales N(0,1) selon le mode RNG choisi.

        Paramètres
        ----------
        size : int
            Nombre de tirages.

        Retourne
        --------
        np.ndarray
            Tableau de N(0,1) de taille `size`.
        """
        if self._use_legacy:
            # RNG global NumPy
            return np.random.normal(size=size)
        return self._rng.normal(size=size)

    # -------------------------
    # Tirage exact (x_t, y_t)
    # -------------------------

    def _simulate_xy(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulation exacte du couple gaussien (x_t, y_t) sous la mesure risque-neutre Q.

        Dynamique factors (G2++) :
          x_t = sigma ∫_0^t e^{-a(t-s)} dW1(s)
          y_t = eta   ∫_0^t e^{-b(t-s)} dW2(s)
          corr(dW1, dW2) = rho

        On utilise ici le fait que (x_t, y_t) est un vecteur gaussien centré
        dont on connaît variance/covariance en forme fermée.

        Paramètres
        ----------
        t : float
            Horizon de simulation (en années).

        Retourne
        --------
        (x, y) : tuple[np.ndarray, np.ndarray]
            Deux tableaux (n_paths,) correspondant aux tirages de x_t et y_t.
        """
        t = float(t)

        # À t ~ 0 : pas de variance => facteurs nuls
        if t <= 1e-16:
            x = np.zeros(self.n_paths)
            y = np.zeros(self.n_paths)
            return x, y

        # Lecture des paramètres du modèle
        a = float(self.model.parameters["a"])
        b = float(self.model.parameters["b"])
        rho = float(self.model.parameters["rho"])
        sigma = float(self.model.parameters["sigma"])
        eta = float(self.model.parameters["eta"])

        # Variances et covariance fermées pour OU gaussiens corrélés
        vx = (sigma**2 / (2.0 * a)) * (1.0 - np.exp(-2.0 * a * t))
        vy = (eta**2 / (2.0 * b)) * (1.0 - np.exp(-2.0 * b * t))
        cxy = (rho * sigma * eta / (a + b)) * (1.0 - np.exp(-(a + b) * t))

        # Tirage de deux normales indépendantes
        z1 = self._normal(self.n_paths)
        z2 = self._normal(self.n_paths)

        # Construction stable d'un couple corrélé via une factorisation 2x2 :
        # x = sx * z1
        # y = alpha * z1 + beta * z2
        sx = np.sqrt(max(vx, 0.0))
        alpha = cxy / (sx + 1e-18)  
        beta2 = vy - alpha**2       
        beta = np.sqrt(max(beta2, 0.0))

        x = sx * z1
        y = alpha * z1 + beta * z2
        return x, y

    # -------------------------
    # Distribution du ZC bond
    # -------------------------

    def zero_coupon_bond(self, t: float, T: float) -> np.ndarray:
        """
        Distribution de P(t,T) sur n_paths scénarios.

        Forme affine gaussienne (sous G2++) :
          P(t,T) = P(0,T)/P(0,t) * exp( -B_a(t,T) x_t - B_b(t,T) y_t - 0.5 v^2(t,T) )

        - Le ratio P(0,T)/P(0,t) vient de l’ajustement à la courbe initiale.
        - B_a, B_b sont les loadings déterministes des facteurs.
        - Ici v^2(t,T) est pris via model.v2_caplet(expiry=t, maturity=T)

        Paramètres
        ----------
        t : float
            Temps d’évaluation (en années).
        T : float
            Maturité du ZC bond (doit vérifier T >= t).

        Retourne
        --------
        np.ndarray
            Tableau (n_paths,) des prix simulés P(t,T).
.
        """
        t = float(t)
        T = float(T)

        # Vérification : on ne peut pas demander P(t,T) avec T < t
        if T < t - 1e-12:
            raise ValueError("Doit avoir T >= t pour P(t,T).")

        # Si T == t : P(t,t)=1 (ZC bond qui mature instantanément)
        if abs(T - t) < 1e-12:
            return np.ones(self.n_paths)

        # Ratio d'ajustement à la courbe initiale (fit exact à t=0)
        ratio = float(self.curve.discount(T)) / (float(self.curve.discount(t)) + 1e-18)

        # Tirage exact des facteurs (x_t, y_t)
        x, y = self._simulate_xy(t)

        # Loadings déterministes
        Ba = float(self.model.B_a(t, T))
        Bb = float(self.model.B_b(t, T))

        # Terme de variance intégrée (ajustement convexité)
        v2 = float(self.model.v2_caplet(t, T))
        adj = -0.5 * v2

        # Formule affine : ratio * exp( -Ba x_t - Bb y_t + adj )
        return ratio * np.exp(-Ba * x - Bb * y + adj)
