import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from ir.models.hw1f import HullWhiteCurveBuilder
from ir.models.hw2f import HullWhite2FModel


class HullWhitePricer:
    """
    Moteur de pricing de produits de taux sous Hull–White 1 facteur,
    basé sur une unique instance de HullWhiteCurveBuilder.

    Produits supportés
    -----------------------------------
    - Options sur obligations zéro-coupon (calls & puts)
    - Caplets / Caps et floorlets / Floors (via décomposition en options ZC)
    - Swaps et swaptions (via Jamshidian)
    - Obligations à coupon, FRN, options sur obligations (Jamshidian)

    Attributs
    ---------
    curve_sim : HullWhiteCurveBuilder
        Objet englobant :
          - le modèle HW1F (formules fermées)
          - un moteur de simulation 
          - la courbe d’actualisation
    model : HullWhiteModel
        Modèle HW1F (accès à A(t,T), B(t,T), discount_factor, etc.)
    curve : Curve
        Courbe d’actualisation initiale
    """

    def __init__(self, curve, n_paths=10**5, n_steps=252, seed=2025, hw_params=None):
        """
        Initialise le pricer HW1F.

        Paramètres
        ----------
        curve : Curve
            Courbe d’actualisation P(0,t).
        n_paths : int
            Nombre de scénarios Monte Carlo .
        n_steps : int
        seed : int
            Graine RNG.
        hw_params : dict | None
            Paramètres HW1F : {'a', 'sigma', 'r0'}.
        """
        self.curve = curve

        # CurveBuilder encapsule le modèle HW1F et l'éventuelle simulation
        self.curve_sim = HullWhiteCurveBuilder(
            curve,
            params=hw_params,
            n_paths=n_paths,
            n_steps=n_steps,
            seed=seed,
        )

        self.model = self.curve_sim.model

    def set_simulation(self, n_paths=None, n_steps=None, seed=None):
        """
        Met à jour les paramètres de simulation Monte Carlo *après* l'initialisation.

        Remarques
        ---------
        - Ici, on modifie directement l'objet self.curve_sim.sim.
        - Le reseed se fait via np.random.seed(seed), ce qui affecte le RNG global numpy.

        Paramètres
        ----------
        n_paths : int, optionnel
            Nouveau nombre de scénarios.
        n_steps : int, optionnel
            Nouveau nombre de pas.
        seed : int, optionnel
            Nouvelle graine RNG.
        """
        if n_paths is not None:
            self.curve_sim.sim.n_paths = int(n_paths)
        if n_steps is not None:
            self.curve_sim.sim.n_steps = int(n_steps)
        if seed is not None:
            np.random.seed(seed)

    def zero_bond_put(self, T, S, K, mc=False):
        """
        Prix d'un put européen sur une obligation zéro-coupon P(T,S).

        Contexte
        --------
        Sous HW1F, l'option sur ZC bond a une formule fermée (type Black sur ZC),
        avec une volatilité effective sigma_p dépendant de (a, sigma, T, S).

        Paramètres
        ----------
        T : float
            Expiry de l'option (en années).
        S : float
            Maturité du ZC bond (S > T).
        K : float
            Strike (prix) de l'obligation P(T,S).
        mc : bool

        Retourne
        --------
        float
            PV du put.
        """
        # Cas limite : option qui expire immédiatement
        if T == 0:
            P_0S = self.model.discount_factor(S)
            return max(K - P_0S, 0)

        sigma = self.model.parameters["sigma"]
        a = self.model.parameters["a"]

        # B(T,S) intervient dans la volatilité effective du bond
        B = self.model.B(T, S)

        P_S = self.model.discount_factor(S)
        P_T = self.model.discount_factor(T)

        # Vol effective du log prix du bond (formule standard HW)
        sigma_p = sigma * np.sqrt((1 - np.exp(-2 * a * T)) / (2 * a)) * B

        # Paramètre h de la formule fermée
        h = (1 / sigma_p) * np.log(P_S / (K * P_T)) + 0.5 * sigma_p

        # Formule fermée du put sur ZC bond
        V0 = K * P_T * norm.cdf(-h + sigma_p) - P_S * norm.cdf(-h)
        return V0

    def zero_bond_call(self, T, S, K):
        """
        Prix d'un call européen sur une obligation zéro-coupon P(T,S).

        Paramètres
        ----------
        T : float
            Expiry.
        S : float
            Maturité du bond (S > T).
        K : float
            Strike (prix) du bond.

        Retourne
        --------
        float
            PV du call.
        """
        sigma = self.model.parameters["sigma"]
        a = self.model.parameters["a"]

        B = self.model.B(T, S)
        P_S = self.model.discount_factor(S)
        P_T = self.model.discount_factor(T)

        sigma_p = sigma * np.sqrt((1 - np.exp(-2 * a * T)) / (2 * a)) * B
        h = (1 / sigma_p) * np.log(P_S / (K * P_T)) + 0.5 * sigma_p

        V0 = P_S * norm.cdf(h) - K * P_T * norm.cdf(h - sigma_p)
        return V0

    def caplet(self, T1, T2, N, K, method="js"):
        """
        Prix d'un caplet (sur taux simple) en utilisant la représentation via option sur ZC bond.

        Notations
        ---------
        - Fixing : T1
        - Paiement : T2
        - Delta = T2 - T1
        - Strike "bond" : K_bond = 1 + K*Delta
          (relation entre payoff caplet et prix du ZC bond)

        Paramètres
        ----------
        T1 : float
            Date de fixing.
        T2 : float
            Date de paiement (T2 > T1).
        N : float
            Notional.
        K : float
            Strike du caplet (taux).
        method : str
            - 'js' : méthode Jamshidian via zéro-bond put
            - 'cf' : formule fermée équivalente (directe)

        Retourne
        --------
        float
            PV du caplet.
        """
        Delta = T2 - T1
        K_bond = 1 + K * Delta

        if method == "js":
            # Caplet = N * (1+KΔ) * Put sur ZC bond de strike 1/(1+KΔ)
            put_price = self.zero_bond_put(T1, T2, 1 / K_bond)
            Caplet = K_bond * put_price

        elif method == "cf":
            # Variante "closed-form" explicitée : même résultat en théorie
            sigma = self.model.parameters["sigma"]
            a = self.model.parameters["a"]
            B = self.model.B(T1, T2)
            P_T2 = self.model.discount_factor(T2)
            P_T1 = self.model.discount_factor(T1)

            sigma_p = sigma * np.sqrt((1 - np.exp(-2 * a * T1)) / (2 * a)) * B
            h = (1 / sigma_p) * np.log(P_T2 * K_bond / P_T1) + 0.5 * sigma_p

            Caplet = (P_T1 * norm.cdf(-h + sigma_p) - K_bond * P_T2 * norm.cdf(-h))

        else:
            raise ValueError("method must be 'js' or 'cf'.")

        return N * Caplet

    def cap(self, Tau, N, K):
        """
        Prix d'un cap comme somme de caplets.

        Paramètres
        ----------
        Tau : list[float]
            Dates : [T0, T1, ..., Tn] où chaque caplet est sur (T_{i-1}, T_i).
        N : float
            Notional.
        K : float
            Strike (taux).

        Retourne
        --------
        float
            PV du cap.
        """
        Cap = 0.0

        # Somme des caplets, i=1..n
        for i in range(1, len(Tau)):
            t_prev = Tau[i - 1]
            t_curr = Tau[i]
            Delta = t_curr - t_prev
            K_bond = 1 + K * Delta

            # Put ZC bond par Jamshidian
            put_price = self.zero_bond_put(t_prev, t_curr, 1 / K_bond)
            Cap += K_bond * put_price

        return N * Cap

    def floor(self, Tau, N, K, mc=False):
        """
        Prix d'un floor comme somme de floorlets.

        Dans la représentation ZC bond, un floorlet correspond à un call sur ZC bond.

        Paramètres
        ----------
        Tau : list[float]
            Dates : [T0, T1, ..., Tn].
        N : float
            Notional.
        K : float
            Strike (taux).
        mc : bool

        Retourne
        --------
        float
            PV du floor.
        """
        Floor = 0.0

        for i in range(1, len(Tau)):
            t_prev = Tau[i - 1]
            t_curr = Tau[i]
            Delta = t_curr - t_prev
            K_bond = 1 + K * Delta

            # Call sur ZC bond par Jamshidian
            call_price = self.zero_bond_call(t_prev, t_curr, 1 / K_bond)
            Floor += K_bond * call_price

        return N * Floor

    def swap(self, Tau, N, K, payer=True, mc=False):
        """
        Prix d'un swap vanilla (jambe fixe vs jambe flottante) sous la courbe initiale.

        Convention
        ----------
        PV = N * w * (PV_float - PV_fixed)
        où :
          - w = +1 pour payer swap (payer fixe, recevoir flottant)
          - w = -1 pour receiver swap

        Paramètres
        ----------
        Tau : list[float]
            Dates de paiement jambe fixe : [T0, T1, ..., Tn].
        N : float
            Notional.
        K : float
            Taux fixe.
        payer : bool
            True = payer swap ; False = receiver swap.
        mc : bool

        Retourne
        --------
        float
            PV du swap.
        """
        w = 1 if payer else -1

        # Calcul annuité A0 = sum Δ_i P(0,Ti)
        Annuity = 0.0
        for i in range(1, len(Tau)):
            Delta = Tau[i] - Tau[i - 1]
            P_T = self.model.discount_factor(Tau[i])
            Annuity += Delta * P_T

        # Jambe fixe
        Fixed_leg = Annuity * K

        # Jambe flottante (swap par swap-parity) : P(0,T0) - P(0,Tn)
        Floating_leg = self.model.discount_factor(Tau[0]) - self.model.discount_factor(Tau[-1])

        Swap = N * w * (Floating_leg - Fixed_leg)
        return Swap

    def swaption(self, Tau, N, K, payer=True, mc=False):
        """
        Prix d'une swaption européenne via décomposition de Jamshidian.

        Principe (HW1F)
        --------------
        Le swaption payoff peut être décomposé en portefeuille d'options sur ZC bonds,
        après avoir trouvé le taux court critique r* (solution d'une équation).

        Paramètres
        ----------
        Tau : list[float]
            Dates de paiement jambe fixe : [T0 (= expiry), T1, ..., Tn].
        N : float
            Notional.
        K : float
            Taux fixe (strike).
        payer : bool
            True = payer swaption ; False = receiver swaption.
        mc : bool

        Retourne
        --------
        float
            PV de la swaption.
        """
        # w sert seulement au payoff (mais ici on s'en sert pour sélectionner put/call)
        w = 1 if payer else -1

        T = Tau[0]   # Expiry
        S = Tau[-1]  # Maturité dernière date

        # Étape 1 : trouver r* (taux court critique)
        r_star = self._find_rstar(T, Tau, K)

        # Étape 2 : jambe fixe => somme d'options ZC bond sur chaque date de paiement
        fixed_leg = 0.0
        for i in range(1, len(Tau)):
            T1 = Tau[i - 1]
            T2 = Tau[i]
            Delta = T2 - T1

            # Prix du ZC bond à T pour maturité T2 : P(T,T2) = A(T,T2)*exp(-B(T,T2)*r(T))
            # Jamshidian : on construit un strike K_i = P(T,T2; r*)
            B = self.model.B(T, T2)
            A = self.model.A(T, T2)
            K_i = A * np.exp(-B * r_star)

            # Payer swaption -> puts ; Receiver -> calls (sur ZC bonds)
            option = self.zero_bond_put(T, T2, K_i) if payer else self.zero_bond_call(T, T2, K_i)
            fixed_leg += Delta * K * option

        # Étape 3 : jambe flottante => option sur le ZC bond maturing at S (dernier DF)
        B_N = self.model.B(T, S)
        A_N = self.model.A(T, S)
        K_N = A_N * np.exp(-B_N * r_star)

        floating_leg = self.zero_bond_put(T, S, K_N) if payer else self.zero_bond_call(T, S, K_N)

        # PV swaption = N * (floating_leg + fixed_leg)
        swaption = N * (floating_leg + fixed_leg)
        return swaption

    def coupon_bond(self, Tau, C, N):
        """
        Prix d'une obligation à coupon simple, valorisée sur la courbe initiale.

        Paramètres
        ----------
        Tau : list[float]
            Dates de paiement (T1, T2, ..., TN).
        C : float
            Taux de coupon annualisé.
        N : float
            Notional.

        Retourne
        --------
        float
            PV de l'obligation à coupon.
        """
        bond_price = 0.0

        Delta = (Tau[-1] - Tau[0])

        for i in range(len(Tau)):
            P_T = self.curve.discount(Tau[i])

            # Dernier flux = principal + coupon ; sinon coupon seul
            cashflow = N * (1 + C * Delta) if i == len(Tau) - 1 else N * C * Delta
            bond_price += cashflow * P_T

        return bond_price

    def floating_rate_note(self, Tau, N):
        """
        Prix d'une FRN (Floating Rate Note) simple.

        Paramètres
        ----------
        Tau : list[float]
            Dates de paiement.
        N : float
            Notional.

        Retourne
        --------
        float
            PV de la FRN.
        """
        disc_cf = self.swap(Tau, N, K=0, payer=False, mc=False)

        disc_notional = N * self.model.discount_factor(Tau[-1])

        frn_price = disc_cf + disc_notional
        return frn_price

    def bond_option(self, T, Tau, C, K, N, call=True, mc=False):
        """
        Option européenne sur obligation à coupon, via décomposition de Jamshidian.

        Paramètres
        ----------
        T : float
            Expiry de l'option.
        Tau : list[float]
            Dates de paiement des coupons (T1, ..., TN).
        C : float
            Taux de coupon annualisé.
        K : float
            Strike (prix absolu de l'obligation, pas un %).
        N : float
            Notional.
        call : bool
            True = call ; False = put.
        mc : bool

        Retourne
        --------
        float
            PV de l'option sur obligation.
        """
        # Étape 1 : trouver r* tel que Prix_obligation(T; r*) = K
        r_star = self._find_rstar_bond(T, Tau, C, N, K)

        bond_option = 0.0
        Delta = (Tau[-1] - Tau[0])

        # Étape 2 : décomposition en options sur ZC bonds
        for i in range(len(Tau)):
            B = self.model.B(T, Tau[i])
            A = self.model.A(T, Tau[i])
            K_i = A * np.exp(-B * r_star)  # strike du ZC bond P(T,T_i) au taux r*

            # Option sur ZC bond : call ou put selon option globale
            option = self.zero_bond_call(T, Tau[i], K_i) if call else self.zero_bond_put(T, Tau[i], K_i)

            # Cashflow associé à la date Tau[i]
            cashflow = N * (1 + C * Delta) if i == len(Tau) - 1 else N * C * Delta
            bond_option += cashflow * option

        return bond_option

    # --- Méthodes auxiliaires (Jamshidian decomposition) --- #

    def _jamshidian_root(self, T, Tau, K, r_star):
        """
        Fonction de root-finding pour la swaption (Jamshidian).

        On cherche r* tel que :
          sum_i Δ_i K P(T, T_i; r*) = 1 - P(T, T_n; r*)
        (forme équivalente aux conditions de décomposition)

        Paramètres
        ----------
        T : float
            Expiry.
        Tau : list[float]
            Dates de paiement swap.
        K : float
            Strike (taux fixe).
        r_star : float
            Candidat pour r*.

        Retourne
        --------
        float
            Valeur de l'équation (doit être 0 à la racine).
        """
        root = 0.0

        # Partie jambe fixe : sum Δ_i K P(T,T_i; r*)
        for i in range(1, len(Tau)):
            T1 = Tau[i - 1]
            T2 = Tau[i]
            Delta = T2 - T1

            B = self.model.B(T, T2)
            A = self.model.A(T, T2)
            P_i = A * np.exp(-B * r_star)

            root += Delta * K * P_i

        # Attention : ici, P_i est celui du dernier i de la boucle (donc maturité T_n)
        # Root complet : fixe - (1 - P(T,T_n))
        root = root - (1 - P_i)
        return root

    def _find_rstar(self, T, Tau, K, x_min=-3, x_max=3):
        """
        Recherche du taux court critique r* pour la swaption via Brent.

        Paramètres
        ----------
        T : float
            Expiry.
        Tau : list[float]
            Dates de paiement.
        K : float
            Strike (taux fixe).
        x_min, x_max : float
            Bornes de recherche pour brentq.

        Retourne
        --------
        float
            r*.
        """
        f = lambda r: self._jamshidian_root(T, Tau, K, r)
        r_star = brentq(f, x_min, x_max, xtol=1e-12)
        return r_star

    def _jamshidian_root_bond(self, T, Tau, C, N, K_strike, r_star):
        """
        Fonction de root-finding pour option sur obligation à coupon (Jamshidian).

        On cherche r* tel que :
          sum_i cashflow_i * P(T, T_i; r*) = K_strike

        Paramètres
        ----------
        T : float
            Expiry.
        Tau : list[float]
            Dates coupons.
        C : float
            Coupon annualisé.
        N : float
            Notional.
        K_strike : float
            Strike (prix de l'obligation).
        r_star : float
            Candidat pour r*.

        Retourne
        --------
        float
            (Prix obligation à T au taux r*) - K_strike
        """
        bond_price = 0.0
        Delta = (Tau[-1] - Tau[0])

        for i in range(len(Tau)):
            B = self.model.B(T, Tau[i])
            A = self.model.A(T, Tau[i])
            P_i = A * np.exp(-B * r_star)

            cashflow = N * (1 + C * Delta) if i == len(Tau) - 1 else N * C * Delta
            bond_price += cashflow * P_i

        return bond_price - K_strike

    def _find_rstar_bond(self, T, Tau, C, N, K_strike, x_min=-3, x_max=3):
        """
        Recherche de r* pour option sur obligation à coupon via Brent.

        Paramètres
        ----------
        T : float
            Expiry.
        Tau : list[float]
            Dates coupons.
        C : float
            Coupon annualisé.
        N : float
            Notional.
        K_strike : float
            Strike prix.
        x_min, x_max : float
            Bornes de recherche Brent.

        Retourne
        --------
        float
            r*.
        """
        f = lambda r: self._jamshidian_root_bond(T, Tau, C, N, K_strike, r)
        r_star = brentq(f, x_min, x_max, xtol=1e-12)
        return r_star
