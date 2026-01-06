from __future__ import annotations

from scipy.optimize import minimize, brentq
import numpy as np
from scipy.stats import norm
import itertools
from typing import Callable, Optional


class HullWhite2FProfileCalibrator:
    """
    Calibrage "profilé" du modèle Hull–White 2 facteurs (G2++)
    sur des swaptions, via :
      - une approximation gaussienne du taux de swap
      - une valorisation en Bachelier (normal model) sur le taux de swap

    Boucle externe : (a, b, rho)
    Boucle interne : (sigma, eta)

    Données de marché attendues dans market_prices :
        - "Prices": list[float]  (PV ou prime forward, cf. use_forward_premium)
        - "Strike": list[float]  (en % -> sera divisé par 100)
        - "Notional": list[float]
        - "Dates": list[list[float]]  où Dates[i] = [T0, T1, ..., Tn]

    Options
    -------
    use_forward_premium : bool
        - Si True : on compare model_price / DF(T0) au prix marché
          (cohérent avec ta convention 1F en calibration "Swaptions").
        - Si False : on compare directement les PV.

    payer : bool
        Hypothèse "payer swaption" pour tous les instruments, sauf si tu fournis
        explicitement "Payer" dans market_prices.
    """

    def __init__(
        self,
        pricer,
        market_prices,
        use_forward_premium=True,
        progress_cb: Optional[Callable[[dict], None]] = None,
    ):
        # Pricer 2F (contient curve + model + conventions)
        self.pricer = pricer
        self.model = pricer.model
        self.curve = pricer.curve

        # Dictionnaire de données de marché (swaptions)
        self.market_prices = market_prices

        # Convention de comparaison (PV vs prime forward)
        self.use_forward_premium = bool(use_forward_premium)

        # Callback UI optionnel (Streamlit, barre de progrès, etc.)
        self.progress_cb = progress_cb

        # Présence éventuelle d'un flag Payer/Receiver par instrument
        self.has_payer_flags = ("Payer" in market_prices)

        # Historique boucle externe : dicts (a,b,rho,sigma,eta,rmsre)
        self.outer_history = []

        # Historique boucle interne : tuples (sigma, eta, rmsre) pour le candidat outer courant
        self.inner_history = []

        # Pré-calcul des objets "statiques" par instrument (ne dépend pas de a,b,rho,sigma,eta)
        # => permet d'éviter de recalculer Tau, A0, S0, poids c, DF(T0) à chaque évaluation.
        self._instr = self._precompute_instruments()

    # -------------------------
    # Gestion des paramètres du modèle
    # -------------------------

    def _set_params(self, a, b, rho, sigma, eta):
        # Met à jour les paramètres du modèle en float (robuste aux types numpy)
        self.model.parameters["a"] = float(a)
        self.model.parameters["b"] = float(b)
        self.model.parameters["rho"] = float(rho)
        self.model.parameters["sigma"] = float(sigma)
        self.model.parameters["eta"] = float(eta)

    # -------------------------
    # Pré-calcul par swaption
    # -------------------------

    def _annuity_and_swap_rate_0(self, Tau):
        """
        Calcule, avec les mêmes conventions que ton pricer :
          - Delta_i = Tau[i] - Tau[i-1]
          - A0 = somme_i Delta_i * P(0, Ti)   (annuité)
          - S0 = (P(0,T0) - P(0,Tn)) / A0     (taux de swap forward à t=0)

        Retourne
        --------
        (A0, S0) en float
        """
        T0 = float(Tau[0])
        Tn = float(Tau[-1])

        # Annuité A0 (pondération des DF par les accruals)
        A0 = 0.0
        for i in range(1, len(Tau)):
            Ti = float(Tau[i])
            delta = float(Tau[i] - Tau[i - 1])
            A0 += delta * self.curve.discount(Ti)

        # Sécurité : une annuité nulle/negative signale un problème de courbe/dates
        if A0 <= 0:
            raise ValueError("Annuity A0 must be > 0.")

        # Swap rate forward S0 (standard)
        S0 = (self.curve.discount(T0) - self.curve.discount(Tn)) / A0
        return float(A0), float(S0)

    def _frozen_weights(self, Tau, A0, S0):
        """
        Construit les poids "frozen" c associés à la variable :
          S(T) ≈ somme_j c_j * P(T, U_j)
        sur la grille U = Tau.

        Formules (approximation gaussienne du taux de swap) :
          c[T0] += +P(0,T0)/A0
          c[Tn] += -P(0,Tn)/A0
          c[Ti] += -(S0/A0) * Delta_i * P(0,Ti), i=1..n

        Retourne
        --------
        U : list[float]  (copie de Tau en float)
        c : np.ndarray   (poids gelés)
        """
        # U est la grille des dates (en float)
        U = [float(x) for x in Tau]
        m = len(U)

        # Poids c_j
        c = np.zeros(m, dtype=float)

        # Terme "numérateur" du swap rate (P(0,T0) - P(0,Tn))
        c[0] += self.curve.discount(U[0]) / A0
        c[-1] += -self.curve.discount(U[-1]) / A0

        # Terme "annuité" : contribution des cashflows fixes via S0
        for i in range(1, m):
            Ti = U[i]
            delta = U[i] - U[i - 1]
            c[i] += -(S0 / A0) * delta * self.curve.discount(Ti)

        return U, c

    def _precompute_instruments(self):
        """
        Pré-calcul, pour chaque swaption i :
          - Tau, U, c
          - expiry T = Tau[0], DF(T)
          - A0, S0
          - K (en taux décimal), N
          - prix marché mkt
          - flag payer (si fourni)

        Objectif
        --------
        Isoler ce qui ne dépend pas des paramètres HW2F, pour accélérer la calibration.
        """
        prices = self.market_prices["Prices"]
        strikes = self.market_prices["Strike"]
        notionals = self.market_prices["Notional"]
        dates = self.market_prices["Dates"]

        n = len(prices)
        instr = []

        for i in range(n):
            # Dates de paiement du swap sous-jacent
            Tau = [float(x) for x in dates[i]]
            if len(Tau) < 2:
                raise ValueError(f"Instrument {i}: Tau must contain at least [T0, Tn].")

            # Expiry de la swaption = début du swap (T0)
            T = float(Tau[0])
            DF = float(self.curve.discount(T))

            # A0 et S0 à t=0
            A0, S0 = self._annuity_and_swap_rate_0(Tau)

            # Poids frozen (approx swap rate gaussien)
            U, c = self._frozen_weights(Tau, A0, S0)

            # Strike (% -> décimal) + notional + prix marché
            K = float(strikes[i]) / 100.0
            N = float(notionals[i])
            mkt = float(prices[i])

            # Payer/Receiver (par défaut payer)
            payer = True
            if self.has_payer_flags:
                payer = bool(self.market_prices["Payer"][i])

            instr.append(
                {
                    "Tau": Tau,
                    "U": U,
                    "c": c,
                    "T": T,
                    "DF": DF,
                    "A0": A0,
                    "S0": S0,
                    "K": K,
                    "N": N,
                    "mkt": mkt,
                    "payer": payer,
                }
            )

        return instr

    # -------------------------
    # Quantités dépendantes de la boucle externe (a, b)
    # -------------------------

    def _compute_Qs_for_outer(self, a, b):
        """
        Pour un couple (a, b) donné (boucle externe), calcule pour chaque instrument :
            Qaa = c^T I_aa c
            Qbb = c^T I_bb c
            Qab = c^T I_ab c

        Ces Q sont des “matrices quadratiques compressées” qui capturent la variance/covariance
        des log-bonds dans l'approximation gaussienne du taux de swap.

        Détails
        -------
        - I_aa, I_bb, I_ab sont des intégrales fermées (closed-form) en HW2F,
          implémentées dans HullWhite2FModel.
        - Une fois Qaa/Qbb/Qab connus, la valorisation est très rapide pour (sigma, eta).
        """
        # Tableaux Q par instrument
        Qaa = np.zeros(len(self._instr), dtype=float)
        Qbb = np.zeros(len(self._instr), dtype=float)
        Qab = np.zeros(len(self._instr), dtype=float)

        # Récupération des fonctions d'intégrales (selon où elles sont définies)
        I_aa = self.model.__class__.I_aa if hasattr(self.model.__class__, "I_aa") else None
        I_bb = self.model.__class__.I_bb if hasattr(self.model.__class__, "I_bb") else None
        I_ab = self.model.__class__.I_ab if hasattr(self.model.__class__, "I_ab") else None

        # Fallback si jamais le modèle courant ne porte pas ces méthodes
        if I_aa is None or I_bb is None or I_ab is None:
            from models.hw2f import HullWhite2FModel
            I_aa = HullWhite2FModel.I_aa
            I_bb = HullWhite2FModel.I_bb
            I_ab = HullWhite2FModel.I_ab

        # Pour chaque instrument, on calcule la forme quadratique c^T I c
        for k, ins in enumerate(self._instr):
            T = ins["T"]   # expiry
            U = ins["U"]   # grille dates
            c = ins["c"]   # poids frozen

            qaa = 0.0
            qbb = 0.0
            qab = 0.0

            # Double somme (i,j) sur les dates U
            for i, Ui in enumerate(U):
                ci = c[i]
                if ci == 0.0:
                    continue
                for j, Uj in enumerate(U):
                    cj = c[j]
                    if cj == 0.0:
                        continue
                    # Intégrales fermées HW2F
                    qaa += ci * cj * I_aa(T, Ui, Uj, a)
                    qbb += ci * cj * I_bb(T, Ui, Uj, b)
                    qab += ci * cj * I_ab(T, Ui, Uj, a, b)

            # Stockage pour l'instrument k
            Qaa[k] = qaa
            Qbb[k] = qbb
            Qab[k] = qab

        return Qaa, Qbb, Qab

    # -------------------------
    # Objectif interne (sigma, eta)
    # -------------------------

    def _price_swaption_from_Qs(self, ins, rho, sigma, eta, qaa, qbb, qab):
        """
        Pricing rapide d'une swaption en utilisant Qaa/Qbb/Qab pré-calculés :

          Var[S(T)] = sigma^2 * qaa + eta^2 * qbb + 2 * rho * sigma * eta * qab

        Puis valorisation Bachelier sur le taux de swap :
          PV = N * A0 * BachelierPrice(S0, K, sigma_N, T)

        Remarques
        ---------
        - On travaille sur la variance du taux de swap à l'échéance T, puis on en déduit
          une volatilité normale (sigma_N).
        - payer/receiver géré via w = +1 (payer) ou -1 (receiver).
        """
        T = ins["T"]
        A0 = ins["A0"]
        S0 = ins["S0"]
        K = ins["K"]
        N = ins["N"]
        payer = ins["payer"]

        # Variance du taux de swap à l'échéance (clamp à 0 pour stabilité numérique)
        varS = (sigma * sigma) * qaa + (eta * eta) * qbb + 2.0 * rho * sigma * eta * qab
        varS = float(max(varS, 0.0))

        # Signe pour payer/receiver (payer = +, receiver = -)
        w = 1.0 if payer else -1.0

        # Cas T <= 0 : payoff immédiat (théorique, rare en calibration)
        if T <= 0:
            return float(N * A0 * max(w * (S0 - K), 0.0))

        # Cas de variance quasi nulle : on retombe sur l'intrinsèque
        if varS < 1e-30:
            return float(N * A0 * max(w * (S0 - K), 0.0))

        # Volatilité normale sur le taux (approx) : sigma_N = sqrt(Var / T)
        sigmaN = np.sqrt(varS / T)

        # d de Bachelier : (S0-K)/(sigmaN*sqrt(T))
        d = (S0 - K) / (sigmaN * np.sqrt(T))

        # Formule Bachelier (normal model)
        price = N * A0 * (w * (S0 - K) * norm.cdf(w * d) + sigmaN * np.sqrt(T) * norm.pdf(d))
        return float(price)

    def _inner_objective(self, x, rho, Qaa, Qbb, Qab):
        """
        Fonction objectif interne, où x = (log(sigma), log(eta)).

        - On fixe (a,b,rho) dans la boucle externe
        - On calibre (sigma, eta) en minimisant la RMSRE
        - Selon use_forward_premium :
            * True  => compare PV/DF(T0) à mkt
            * False => compare PV à mkt
        """
        # Paramétrisation log pour garantir sigma>0 et eta>0
        sigma = float(np.exp(x[0]))
        eta = float(np.exp(x[1]))

        eps = 1e-6
        n = len(self._instr)

        # Accumulation RMSRE
        err = 0.0
        for k, ins in enumerate(self._instr):
            # PV modèle (Bachelier sur taux de swap)
            model_pv = self._price_swaption_from_Qs(ins, rho, sigma, eta, Qaa[k], Qbb[k], Qab[k])

            # Conversion PV -> prime forward si demandé
            if self.use_forward_premium:
                model_val = model_pv / (ins["DF"] + 1e-18)
            else:
                model_val = model_pv

            # Prix marché associé
            mkt = ins["mkt"]

            # Erreur relative au carré (stabilisée)
            err += (1.0 / n) * ((model_val - mkt) ** 2) / (mkt * mkt + eps)

        rmsre = float(np.sqrt(err))

        # Historise pour debug / verbose
        self.inner_history.append((sigma, eta, rmsre))

        return rmsre

    def _inner_callback(self, x):
        # Callback interne (optionnel) : affiche l’avancement de la calibration (sigma, eta)
        # Utilisé seulement si verbose_inner=True dans calibrate_profile()
        if not self.inner_history:
            return
        sigma, eta, err = self.inner_history[-1]
        print(f"    sigma: {sigma:.6f}, eta: {eta:.6f}, RMSRE: {err:.5e}")

    def _run_inner_calibration(
        self,
        rho,
        Qaa,
        Qbb,
        Qab,
        init_sigma=0.01,
        init_eta=0.008,
        bounds_sigma=(1e-4, 0.5),
        bounds_eta=(1e-4, 0.5),
        method="L-BFGS-B",
        ftol=1e-6,
        verbose=False,
    ):
        """
        Lance la calibration interne de (sigma, eta) pour un rho fixé
        et des Q pré-calculés (donc pour un (a,b) fixé).

        Retourne
        --------
        (res, sigma_opt, eta_opt)
          - res : OptimizeResult SciPy
          - sigma_opt, eta_opt : paramètres optimaux (reconvertis depuis log-space)
        """
        # Reset de l'historique interne à chaque candidat outer
        self.inner_history = []

        # Point de départ en log-space
        x0 = np.log([init_sigma, init_eta])

        # Bornes en log-space
        bounds = [
            (np.log(bounds_sigma[0]), np.log(bounds_sigma[1])),
            (np.log(bounds_eta[0]), np.log(bounds_eta[1])),
        ]

        # Callback interne seulement en mode verbose (sinon None = pas d'affichage)
        cb = self._inner_callback if verbose else None

        # Optimisation SciPy : minimise la RMSRE interne
        res = minimize(
            lambda x: self._inner_objective(x, rho, Qaa, Qbb, Qab),
            x0,
            bounds=bounds,
            method=method,
            callback=cb,
            options={"ftol": ftol},
        )

        # Reconvertit les paramètres optimaux depuis log-space
        sigma_opt = float(np.exp(res.x[0]))
        eta_opt = float(np.exp(res.x[1]))
        return res, sigma_opt, eta_opt

    # -------------------------
    # API publique : calibrage profilé
    # -------------------------

    def calibrate_profile(
        self,
        # grilles boucle externe
        grid_a=None,
        grid_b=None,
        grid_rho=None,
        # init/bornes boucle interne
        init_sigma=0.01,
        init_eta=0.008,
        bounds_sigma=(1e-4, 0.5),
        bounds_eta=(1e-4, 0.5),
        # settings optim interne
        inner_method="L-BFGS-B",
        inner_ftol=1e-6,
        verbose_inner=False,
        top_k=3,
    ):
        """
        Lance le calibrage profilé :
          1) On parcourt une grille de (a,b,rho)
          2) Pour chaque triplet, on calcule Qaa/Qbb/Qab
          3) On calibre (sigma, eta) en interne
          4) On conserve le meilleur candidat (RMSRE minimale)

        Retourne
        -------
        dict :
          - "best"    : meilleur candidat (paramètres + rmsre + inner_result)
          - "ranking" : liste triée de tous les candidats
        """
        # Grilles par défaut si non fournies (pédagogiques / robustes)
        if grid_a is None:
            grid_a = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30]
        if grid_b is None:
            grid_b = [0.001, 0.003, 0.01, 0.02, 0.05, 0.10]
        if grid_rho is None:
            grid_rho = [-0.95, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4]

        # Contrainte d’identification fréquente : b < a
        outer_candidates = []
        for a, b, rho in itertools.product(grid_a, grid_b, grid_rho):
            if b < a:
                outer_candidates.append((float(a), float(b), float(rho)))

        # Si la contrainte b<a vide la grille, c’est un problème de choix de grilles
        if not outer_candidates:
            raise ValueError("Empty outer grid after applying constraint b < a.")

        outer_total = len(outer_candidates)

        # Messages de contexte (log console)
        print(f"Calibration profilé sur {len(self._instr)} swaptions.")
        print(f"Candidats grille externe : {outer_total}")

        results = []
        best_rmsre = float("inf")

        # Boucle externe : évalue chaque triplet (a,b,rho)
        for idx, (a, b, rho) in enumerate(outer_candidates, start=1):
            # --- progress: début d’un candidat ---
            if self.progress_cb is not None:
                try:
                    self.progress_cb(
                        {
                            "stage": "outer_start",
                            "outer_idx": int(idx),
                            "outer_total": int(outer_total),
                            "a": float(a),
                            "b": float(b),
                            "rho": float(rho),
                            "best_rmsre": None if best_rmsre == float("inf") else float(best_rmsre),
                        }
                    )
                except Exception:
                    # Ne jamais casser la calibration à cause du callback UI
                    pass

            # Affichage du candidat courant
            print(f"\n[Outer {idx}/{outer_total}] a={a:.4f}, b={b:.4f}, rho={rho:+.2f}")

            # Mise à jour des paramètres "outer" dans le modèle (sigma,eta seront calibrés ensuite)
            self.model.parameters["a"] = a
            self.model.parameters["b"] = b
            self.model.parameters["rho"] = rho

            # Calcul Qaa/Qbb/Qab (dépend uniquement de a et b)
            Qaa, Qbb, Qab = self._compute_Qs_for_outer(a, b)

            # Calibration interne (sigma,eta) avec rho fixé + Q's fixés
            res_in, sigma_opt, eta_opt = self._run_inner_calibration(
                rho=rho,
                Qaa=Qaa,
                Qbb=Qbb,
                Qab=Qab,
                init_sigma=init_sigma,
                init_eta=init_eta,
                bounds_sigma=bounds_sigma,
                bounds_eta=bounds_eta,
                method=inner_method,
                ftol=inner_ftol,
                verbose=verbose_inner,
            )

            # Résultat interne : RMSRE associée au candidat outer
            rmsre = float(res_in.fun)
            print(f"  -> inner best: sigma={sigma_opt:.6f}, eta={eta_opt:.6f}, RMSRE={rmsre:.5e}")

            # Mise à jour du meilleur à date
            improved = rmsre < best_rmsre
            if improved:
                best_rmsre = rmsre

            # --- progress: fin d’un candidat ---
            if self.progress_cb is not None:
                try:
                    self.progress_cb(
                        {
                            "stage": "outer_done",
                            "outer_idx": int(idx),
                            "outer_total": int(outer_total),
                            "a": float(a),
                            "b": float(b),
                            "rho": float(rho),
                            "cand_rmsre": float(rmsre),
                            "best_rmsre": float(best_rmsre),
                            "improved": bool(improved),
                        }
                    )
                except Exception:
                    pass

            # Stocke le candidat complet (utile pour ranking + top_k)
            results.append(
                {
                    "a": a,
                    "b": b,
                    "rho": rho,
                    "sigma": sigma_opt,
                    "eta": eta_opt,
                    "rmsre": rmsre,
                    "inner_result": res_in,
                }
            )

        # Trie tous les candidats par erreur croissante
        results.sort(key=lambda d: d["rmsre"])
        best = results[0]

        # Fixe les paramètres finaux (best) dans le modèle
        self._set_params(best["a"], best["b"], best["rho"], best["sigma"], best["eta"])

        # Résumé final console
        print("\n=== Résultat calibration profilé (best) ===")
        print(f"RMSRE: {best['rmsre']:>+8.3%}")
        print("Paramètres:")
        print(f"  a    : {best['a']:.6f}")
        print(f"  b    : {best['b']:.6f}")
        print(f"  rho  : {best['rho']:+.6f}")
        print(f"  sigma: {best['sigma']:.6f}")
        print(f"  eta  : {best['eta']:.6f}")

        print("\nRapport par instrument:")

        # Recalcule Q's pour le meilleur (a,b) afin de produire le report
        Qaa_best, Qbb_best, Qab_best = self._compute_Qs_for_outer(best["a"], best["b"])

        rho = best["rho"]
        sigma = best["sigma"]
        eta = best["eta"]

        # Rapport instrument par instrument : compare modèle vs marché
        for k, ins in enumerate(self._instr):
            model_pv = self._price_swaption_from_Qs(ins, rho, sigma, eta, Qaa_best[k], Qbb_best[k], Qab_best[k])

            # Applique la même convention que dans l’objectif (PV vs prime forward)
            if self.use_forward_premium:
                model_val = model_pv / (ins["DF"] + 1e-18)
            else:
                model_val = model_pv

            mkt = ins["mkt"]
            dif = model_val / (mkt + 1e-12) - 1.0

            Tau = ins["Tau"]
            print(
                f"Swaption {k:>3}: {Tau[0]:>5.2f}Y à {Tau[-1]:<5.2f}Y | "
                f"Modèle: {model_val:>10.4f} | Marché: {mkt:>10.4f} | Diff: {dif:>+8.3%}"
            )

        # Affiche un mini-classement des top_k meilleurs candidats 
        if top_k and top_k > 1:
            print(f"\nTop {min(top_k, len(results))} candidats:")
            for j in range(min(top_k, len(results))):
                r = results[j]
                print(
                    f"  {j+1:>2}. a={r['a']:.4f}, b={r['b']:.4f}, rho={r['rho']:+.2f} | "
                    f"sigma={r['sigma']:.5f}, eta={r['eta']:.5f} | RMSRE={r['rmsre']:.5e}"
                )

        # Retourne le meilleur + le ranking complet 
        return {"best": best, "ranking": results}
