import pandas as pd
import numpy as np


class HullWhiteModel:
    """
    Modèle de taux court Hull–White 1 facteur

    Cette classe implémente les formules analytiques du modèle de Vasicek étendu / Hull–White :
        dr(t) = a * (θ(t) - r(t)) dt + σ dW(t)
    où :
        a     = vitesse de retour à la moyenne (mean reversion)
        σ     = volatilité du taux court
        θ(t)  = drift déterministe (dépendant du temps) ajusté pour coller à la courbe initiale

    Attributs
    ---------
    curve : Curve
        Instance représentant la courbe d'actualisation initiale P(0,T).
    parameters : dict
        Dictionnaire de paramètres du modèle :
            - 'a' : float, vitesse de retour à la moyenne
            - 'sigma' : float, volatilité
            - 'r0' : float, taux court initial
    """

    def __init__(self, curve, parameters=None):
        """
        Initialise le modèle Hull–White à partir d'une courbe de discount et d'un jeu de paramètres.

        Paramètres
        ----------
        curve : Curve
            Courbe d'actualisation utilisée pour obtenir P(0,t) et f(0,t).
        parameters : dict, optionnel
            Dictionnaire optionnel avec les clés 'a', 'sigma', 'r0'.
            Si None, on utilise des valeurs par défaut :
              - a = 0.01
              - sigma = 0.01
              - r0 = curve.inst_forward_rate(0)
        """
        self.curve = curve

        # Valeurs par défaut : permettent un "quick start" sans calibration
        defaults = {"a": 0.01, "sigma": 0.01, "r0": curve.inst_forward_rate(0)}

        if parameters is None:
            parameters = {}

        # Construction du dict final (avec fallback sur defaults si clé absente)
        self.parameters = {
            "a": parameters.get("a", defaults["a"]),
            "sigma": parameters.get("sigma", defaults["sigma"]),
            "r0": parameters.get("r0", defaults["r0"]),
        }

    def inst_forward_rate(self, t):
        """
        Renvoie le taux forward instantané f(0,t) (dérivé de la courbe).

        Paramètres
        ----------
        t : float
            Temps (en années).

        Retourne
        --------
        float
            f(0,t) à la date t.
        """
        # Délégation directe à la courbe
        return self.curve.inst_forward_rate(t)

    def discount_factor(self, t):
        """
        Renvoie le facteur d'actualisation P(0,t) (interpolé via la courbe).

        Paramètres
        ----------
        t : float
            Temps (en années).

        Retourne
        --------
        float
            P(0,t).
        """
        # Délégation directe à la courbe
        return self.curve.discount(t)

    def forward_rate(self, T1, T2):
        """
        Renvoie le taux forward simple F(0; T1, T2) implicite de la courbe.

        Convention (forward simple) :
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
            Taux forward entre T1 et T2.
        """
        return self.curve.forward_rate(T1, T2)

    def alpha(self, t):
        """
        Calcule α(t), la fonction de décalage déterministe dans Hull–White.

        Formule :
            α(t) = f(0,t) + (σ² / (2a²)) * (1 - e^{-a t})²

        Interprétation :
        - α(t) intervient dans l'expression de l'espérance de r(t)
        - elle "force" le modèle à coller à la courbe initiale

        Paramètres
        ----------
        t : float
            Temps (en années).

        Retourne
        --------
        float
            Valeur α(t).
        """
        a = self.parameters["a"]
        sigma = self.parameters["sigma"]
        fwd = self.inst_forward_rate(t)

        return fwd + (sigma**2) / (2 * a**2) * (1 - np.exp(-a * t)) ** 2

    def B(self, t, T):
        """
        Calcule la fonction B(t,T) utilisée dans le pricing des ZC bonds en HW.

        Formule :
            B(t,T) = (1 - e^{-a (T - t)}) / a

        Paramètres
        ----------
        t : float
            Temps courant.
        T : float
            Maturité.

        Retourne
        --------
        float
            Valeur B(t,T).
        """
        a = self.parameters["a"]
        return (1 - np.exp(-a * (T - t))) / a

    def A(self, t, T):
        """
        Calcule la fonction A(t,T) utilisée dans le pricing des obligations zéro-coupon.

        Formule :
            A(t,T) = [P(0,T)/P(0,t)] * exp( B(t,T)*f(0,t)
                     - (σ²/(4a))*(1 - e^{-2at})*B(t,T)² )

        Paramètres
        ----------
        t : float
            Temps courant.
        T : float
            Maturité.

        Retourne
        --------
        float
            Valeur A(t,T).
        """
        a = self.parameters["a"]
        sigma = self.parameters["sigma"]

        # Facteurs d'actualisation initiaux
        P_t = self.discount_factor(t)
        P_T = self.discount_factor(T)

        # Forward instantané à t
        fwd = self.inst_forward_rate(t)

        # Fonction B(t,T)
        B = self.B(t, T)

        # Formule fermée pour A(t,T)
        return (P_T / P_t) * np.exp(
            B * fwd - (sigma**2 / (4 * a)) * (1 - np.exp(-2 * a * t)) * B**2
        )

    def short_rate(self, t, z=None):
        """
        Simule r(t) sous la mesure risque-neutre via la distribution exacte.

        Propriété clé (HW1F gaussian) :
            r(t) ~ Normal( E[r(t)], V[r(t)] )

        Paramètres
        ----------
        t : float
            Temps (en années).
        z : float, optionnel
            Tirage N(0,1). Si None, un tirage est généré.

        Retourne
        --------
        float
            Une réalisation du taux court r(t).
        """
        # Génère un tirage standard normal si pas fourni
        if z is None:
            z = np.random.normal()

        r0 = self.parameters["r0"]
        a = self.parameters["a"]
        sigma = self.parameters["sigma"]

        # Variance exacte de r(t) en HW
        V = (sigma**2 / (2 * a)) * (1 - np.exp(-2 * a * t))

        # Espérance exacte de r(t) 
        # NB : alpha(t) dépend de la courbe initiale
        E = r0 * np.exp(-a * t) + self.alpha(t) - np.exp(-a * t) * self.alpha(0)

        # Construction de r(t) = E + sqrt(V)*z
        return E + np.sqrt(V) * z


class HullWhiteSimulation:
    """
    Moteur de simulation Monte Carlo pour Hull–White 1 facteur.

    Fournit :
      - simulation exacte de r(T) à une maturité 

    Attributs
    ---------
    model : HullWhiteModel
        Instance du modèle HW contenant courbe et paramètres.
    n_paths : int
        Nombre de scénarios Monte Carlo.
    n_steps : int
    seed : int
        Graine aléatoire pour reproductibilité.
    """

    def __init__(self, model: HullWhiteModel, n_paths=10**5, n_steps=100, seed=2025):
        """
        Initialise le moteur de simulation.

        Paramètres
        ----------
        model : HullWhiteModel
            Modèle Hull–White.
        n_paths : int
            Nombre de simulations (scénarios).
        n_steps : int
        seed : int
            Graine RNG pour reproductibilité.
        """
        self.model = model
        self.n_paths = n_paths
        self.n_steps = n_steps

        # Initialise le RNG global numpy 
        self.seed = np.random.seed(seed)

    def simulate_short_rate_direct(self, T):
        """
        Simule r(T) en utilisant la distribution exacte (gaussienne) sous Q.

        Paramètres
        ----------
        T : float
            Horizon en années.

        Retourne
        --------
        ndarray
            Vecteur de taille (n_paths,) contenant les r(T) simulés.
        """
        # Tirages gaussiens indépendants
        z = np.random.normal(size=self.n_paths)

        # Simulation exacte instrument par instrument 
        r = np.array([self.model.short_rate(T, z=z_i) for z_i in z])
        return r


class HullWhiteCurveBuilder:
    """
    "Curve builder" Hull–White : centralise :
      - les formules analytiques (A, B, etc.) via HullWhiteModel
      - des utilitaires Monte Carlo via HullWhiteSimulation
      - un accès pratique à la courbe initiale

    Objectif :
    - offrir une interface simple pour obtenir :
        * r(t) simulé
        * prix de ZC bond P(t,T) (distribution via MC)
    - en s’appuyant sur une courbe P(0,T) déjà construite

    Attributs
    ---------
    model : HullWhiteModel
        Modèle HW construit avec curve + params.
    sim : HullWhiteSimulation
        Moteur MC construit à partir du modèle HW.
    curve : Curve
        Courbe initiale (discount curve).
    """

    def __init__(self, curve, params=None, n_paths=10**5, n_steps=100, seed=2025, smooth=1e-7):
        """
        Initialise le builder HW à partir d'une courbe existante et de paramètres.

        Paramètres
        ----------
        curve : Curve
            Courbe d'actualisation pré-initialisée (times + discount factors).
        params : dict, optionnel
            Paramètres HW : {'a', 'sigma', 'r0'}.
            Si None, defaults (dans HullWhiteModel) sont utilisés.
        n_paths : int
            Nombre de trajectoires/scénarios Monte Carlo.
        n_steps : int
        seed : int
            Graine RNG.
        smooth : float
            Paramètre de lissage 

        Workflow
        --------
        1) Réutilise la curve pour P(0,T) et f(0,t)
        2) Construit HullWhiteModel(curve, params)
        3) Construit HullWhiteSimulation(model, ...)
        """
        self.curve = curve
        self.model = HullWhiteModel(self.curve, params)
        self.sim = HullWhiteSimulation(self.model, n_paths=n_paths, n_steps=n_steps, seed=seed)

    def short_rate(self, t):
        """
        Simule r(t) à une date t via la simulation directe (distribution exacte).

        Paramètres
        ----------
        t : float
            Temps en années.

        Retourne
        --------
        ndarray
            Vecteur (n_paths,) de taux courts simulés.
        """
        return self.sim.simulate_short_rate_direct(t)

    def zero_coupon_bond(self, t, T):
        """
        Calcule la distribution (Monte Carlo) du prix P(t,T) d'un zéro-coupon
        sous la mesure risque-neutre.

        Formule analytique (conditionnelle à r(t)) :
            P(t, T) = A(t, T) * exp(-B(t, T) * r(t))

        Ici :
        - on simule r(t) (distribution sous Q)
        - on applique la formule fermée pour obtenir un échantillon de P(t,T)

        Paramètres
        ----------
        t : float
            Temps courant.
        T : float
            Maturité.

        Retourne
        --------
        ndarray
            Distribution Monte Carlo des prix P(t,T) (taille n_paths).
        """
        # Simule r(t) via distribution exacte
        r_t = self.sim.simulate_short_rate_direct(t)

        # Récupère A(t,T) et B(t,T) du modèle
        A = self.model.A(t, T)
        B = self.model.B(t, T)

        # Prix ZC bond pour chaque scénario
        price = A * np.exp(-B * r_t)
        return price
