from scipy.optimize import minimize, brentq
import numpy as np
from scipy.stats import norm
import itertools
from typing import Callable, Optional, Dict, Any


class HullWhiteCalibrator:
    """
    Calibre les paramètres (a, sigma) du modèle de Hull–White 1 facteur
    à partir de prix de marché de caplets ou de swaptions.

    Notes
    -----
    - Utilise une paramétrisation en log pour imposer la positivité :
      a = exp(x[0]), sigma = exp(x[1]).
    - Fonction objectif : RMSRE (root mean squared relative error) sur les prix
      (ou sur la prime forward pour les swaptions).
    - Conserve des impressions détaillées (callback + rapport final instrument par instrument)

    Patch (progression Streamlit)
    -----------------------------
    - progress_cb : callable optionnel appelé à chaque itération de l’optimiseur (via callback).
      Il reçoit un dict : {"iter": int, "a": float, "sigma": float, "rmsre": float}
    """

    def __init__(
        self,
        pricer,
        market_prices,
        calibrate_to="Caplets",
        progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,  
    ):
        # Pricer HW (contient model + curve + formules caplet/swaption)
        self.pricer = pricer
        self.model = pricer.model

        # Dictionnaire-like de données de marché (strikes, maturités, prix, etc.)
        self.market_prices = market_prices

        # Type de calibration : "Caplets" ou "Swaptions"
        self.calibrate_to = calibrate_to

        # Historique des évaluations de l’objectif : (a, sigma, rmsre)
        # Utile pour debug, graphiques, ou affichage streamlit
        self.history = []

        # Callback optionnel pour faire remonter la progression dans une UI (Streamlit)
        self.progress_cb = progress_cb  

        # Compteur d’itérations basé sur le nombre d’appels au callback
        self._cb_iter = 0  

    # -------- helpers internes -------- #

    def _set_params(self, a: float, sigma: float) -> None:
        # Met à jour les paramètres du modèle dans l’objet pricer.model
        self.model.parameters["a"] = float(a)
        self.model.parameters["sigma"] = float(sigma)

    def _price_instrument(self, i: int) -> float:
        """
        Retourne le prix modèle (ou la prime forward pour les swaptions) pour l’instrument i,
        en utilisant les paramètres courants (a, sigma).
        """
        # Conversion du strike stocké en pourcentage vers un taux décimal (ex: 3% -> 0.03)
        K = self.market_prices["Strike"][i] / 100.0

        # Notional de l’instrument (le pricer attend généralement N explicite)
        N = self.market_prices["Notional"][i]

        if self.calibrate_to == "Caplets":
            # Caplet défini par (Expiry, Maturity) dans ton format de données
            T = self.market_prices["Expiry"][i]
            S = self.market_prices["Maturity"][i]
            return self.pricer.caplet(T, S, N, K)

        elif self.calibrate_to == "Swaptions":
            # Swaption définie par une grille de dates Tau = [T0, T1, ..., Tn]
            Tau = self.market_prices["Dates"][i]

            # Discount factor au début du swap (utile pour convertir en "prime forward")
            DF = self.pricer.curve.discount(Tau[0])

            # Cohérence de convention : vérifier que market_prices['Prices'] est bien défini
            # selon cette convention de "prime forward" (prix / DF).
            return self.pricer.swaption(Tau, N, K) / DF

        # Protection : évite un calibrage silencieux sur un type non supporté
        raise ValueError("Calibration implémentée uniquement pour 'Caplets' et 'Swaptions'.")

    # -------- objectif d’optimisation + callback -------- #

    def objective(self, x):
        """
        Fonction objectif J(x) avec x = (log(a), log(sigma)).
        Retourne la RMSRE sur l’ensemble des instruments.

        RMSRE = sqrt( (1/n) * sum_i ((model_i - mkt_i)^2 / (mkt_i^2 + eps)) )
        """
        # Paramétrisation en log : garantit a>0 et sigma>0
        a = np.exp(x[0])
        sigma = np.exp(x[1])

        # On “pousse” les paramètres dans le modèle avant de price
        self._set_params(a, sigma)

        # Prix de marché (même convention que _price_instrument)
        prices = self.market_prices["Prices"]
        n = len(prices)

        # Terme de stabilisation pour éviter division par ~0 si un prix marché est très petit
        eps = 1e-6

        # Accumulation de l’erreur relative au carré (moyennée)
        err = 0.0
        for i in range(n):
            # Prix observé marché pour l’instrument i
            market_price = prices[i]

            # Prix modèle donné les paramètres courants
            model_price = self._price_instrument(i)

            # Contribution RMSRE : (model-mkt)^2 / (mkt^2 + eps), moyennée sur n
            err += (1.0 / n) * ((model_price - market_price) ** 2) / (market_price**2 + eps)

        # RMSRE finale
        rmsre = float(np.sqrt(err))

        # On stocke l’évaluation : utile pour callback + debug
        self.history.append((a, sigma, rmsre))

        return rmsre

    def callback(self, x):
        """
        Affiche les paramètres (a, sigma) et l’erreur durant l’optimisation
        (comme dans ton callback d’origine).
        Appelle aussi progress_cb si fourni (pour mise à jour live dans l’UI Streamlit).
        """
        # Cas rare : callback appelé avant la première évaluation “history”
        if not self.history:
            return

        # Incrémente le compteur d’itérations "UI"
        self._cb_iter += 1 

        # Dernière valeur évaluée par objective()
        a, sigma, err = self.history[-1]

        # Impression console (capturable via ton capture_stdout)
        print(f"a: {a:.6f}, sigma: {sigma:.6f}, RMSRE: {err:.5e}")

        # On encapsule dans try/except pour ne jamais casser l’optimisation si l’UI plante.
        if self.progress_cb is not None:
            try:
                self.progress_cb(
                    {
                        "iter": int(self._cb_iter),
                        "a": float(a),
                        "sigma": float(sigma),
                        "rmsre": float(err),
                    }
                )
            except Exception:
                # Ne pas casser l’optimisation si la mise à jour UI échoue
                pass

    # -------- point d’entrée principal -------- #

    def calibrate(
        self,
        init_a=0.01,
        init_sigma=0.01,
        bounds_a=(1e-4, 1.0),
        bounds_sigma=(1e-4, 0.5),
        method="L-BFGS-B",
        ftol=1e-6,
    ):
        """
        Lance l’optimisation pour calibrer a et sigma.

        Retourne
        --------
        scipy.optimize.OptimizeResult
        """
        # Réinitialise le compteur d’itérations du callback à chaque run
        self._cb_iter = 0

        # Point de départ : on passe en log-space (compat avec objective)
        x0 = np.log([init_a, init_sigma])

        # Bornes en log-space (cohérent avec x = (log(a), log(sigma)))
        bounds = [
            (np.log(bounds_a[0]), np.log(bounds_a[1])),
            (np.log(bounds_sigma[0]), np.log(bounds_sigma[1])),
        ]

        # Lance l’optimiseur SciPy
        # - objective: calcule RMSRE
        # - callback: affichage + UI
        # - options ftol: tolérance sur la convergence (peut dépendre de la version SciPy)
        result = minimize(
            self.objective,
            x0,
            bounds=bounds,
            method=method,
            callback=self.callback,
            options={"ftol": ftol},
        )

        if result.success:
            # Reconvertit les paramètres optimaux depuis log-space
            a_opt = float(np.exp(result.x[0]))
            sigma_opt = float(np.exp(result.x[1]))

            # Fige les paramètres optimaux dans le modèle
            self._set_params(a_opt, sigma_opt)

            # Résumé console
            print("\nCalibration réussie :")
            print(f"Itérations : {result.nit}")
            print(f"Nombre d’instruments : {len(self.market_prices['Prices'])}")
            print(f"Erreur totale (RMSRE) : {result.fun:>+8.3%}\n")
            print("Paramètres :")
            print(f"a optimal : {a_opt:.6f}")
            print(f"sigma optimal : {sigma_opt:.6f}\n")

            # Rapport instrument par instrument : compare prix modèle vs marché
            for i in range(len(self.market_prices["Prices"])):
                market_price = self.market_prices["Prices"][i]
                model_price = self._price_instrument(i)

                # Différence relative (attention: division stabilisée)
                dif = model_price / (market_price + 1e-12) - 1.0

                if self.calibrate_to == "Caplets":
                    # Détail des dates caplets pour lecture humaine
                    T = self.market_prices["Expiry"][i]
                    S = self.market_prices["Maturity"][i]
                    print(
                        f"Caplet {i:>2}: {T:>5.2f}Y à {S:<5.2f}Y | "
                        f"Modèle: {model_price:>8.2f} | Marché: {market_price:>8.2f} | Écart: {dif:>+8.3%}"
                    )

                elif self.calibrate_to == "Swaptions":
                    # Swaption : affiche seulement début/fin de Tau
                    Tau = self.market_prices["Dates"][i]
                    print(
                        f"Swaption {i:>3}: {Tau[0]:>5.2f}Y à {Tau[-1]:<5.2f}Y | "
                        f"Modèle: {model_price:>8.2f} | Marché: {market_price:>8.2f} | Écart: {dif:>+8.3%}"
                    )

        else:
            # Message d’erreur de l’optimiseur (bornes, non convergence, etc.)
            print("Calibration échouée :", result.message)

        return result
