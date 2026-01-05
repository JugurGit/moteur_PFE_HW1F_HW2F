from __future__ import annotations

from scipy.optimize import minimize, brentq
import numpy as np
from scipy.stats import norm
import itertools
from typing import Callable, Optional


class HullWhite2FProfileCalibrator:
    """
    Profile calibration of Hull-White 2F (G2++) parameters on swaptions
    using the Gaussian swap-rate approximation + Bachelier pricing.

    Outer loop : (a, b, rho)
    Inner loop : (sigma, eta)

    Market data expected in market_prices:
        - "Prices": list[float]  (PV or forward premium, see use_forward_premium)
        - "Strike": list[float]  (in % like your current calibrator, will be /100)
        - "Notional": list[float]
        - "Dates": list[list[float]]  where Dates[i] = [T0, T1, ..., Tn]

    Options
    -------
    use_forward_premium : bool
        If True, compare model_price / DF(T0) to market price (matches your 1F "Swaptions" convention).
        If False, compare PV directly.
    payer : bool
        Assume payer swaption for all instruments, unless you supply "Payer" in market_prices.

    progress_cb : Optional[Callable[[dict], None]]
        If provided, called during the outer loop to report progress.
        Typical payload:
          - stage: "outer_start" | "outer_done"
          - outer_idx, outer_total
          - a, b, rho
          - cand_rmsre (on "outer_done")
          - best_rmsre (best so far)
          - improved (on "outer_done")
    """

    def __init__(
        self,
        pricer,
        market_prices,
        use_forward_premium=True,
        progress_cb: Optional[Callable[[dict], None]] = None,
    ):
        self.pricer = pricer
        self.model = pricer.model
        self.curve = pricer.curve
        self.market_prices = market_prices
        self.use_forward_premium = bool(use_forward_premium)

        # NEW: UI callback (Streamlit progress, etc.)
        self.progress_cb = progress_cb

        # optional per-instrument payer flags
        self.has_payer_flags = ("Payer" in market_prices)

        # histories
        self.outer_history = []  # stores dicts with (a,b,rho,sigma,eta,rmsre)
        self.inner_history = []  # stores tuples (sigma, eta, rmsre) for current outer loop

        # Precompute instrument static objects (do not depend on a,b,rho,sigma,eta)
        self._instr = self._precompute_instruments()

    # -------------------------
    # Model parameter management
    # -------------------------

    def _set_params(self, a, b, rho, sigma, eta):
        self.model.parameters["a"] = float(a)
        self.model.parameters["b"] = float(b)
        self.model.parameters["rho"] = float(rho)
        self.model.parameters["sigma"] = float(sigma)
        self.model.parameters["eta"] = float(eta)

    # -------------------------
    # Precomputation per swaption
    # -------------------------

    def _annuity_and_swap_rate_0(self, Tau):
        """
        Same conventions as your pricer:
          Delta_i = Tau[i] - Tau[i-1]
          A0 = sum Delta_i P(0,Ti)
          S0 = (P(0,T0) - P(0,Tn)) / A0
        """
        T0 = float(Tau[0])
        Tn = float(Tau[-1])

        A0 = 0.0
        for i in range(1, len(Tau)):
            Ti = float(Tau[i])
            delta = float(Tau[i] - Tau[i - 1])
            A0 += delta * self.curve.discount(Ti)

        if A0 <= 0:
            raise ValueError("Annuity A0 must be > 0.")

        S0 = (self.curve.discount(T0) - self.curve.discount(Tn)) / A0
        return float(A0), float(S0)

    def _frozen_weights(self, Tau, A0, S0):
        """
        Build frozen weights c for U = Tau:
          c[T0] += P(0,T0)/A0
          c[Tn] += -P(0,Tn)/A0
          c[Ti] += -(S0/A0) * Delta_i * P(0,Ti), i=1..n
        """
        U = [float(x) for x in Tau]
        m = len(U)
        c = np.zeros(m, dtype=float)

        # numerator
        c[0] += self.curve.discount(U[0]) / A0
        c[-1] += -self.curve.discount(U[-1]) / A0

        # annuity
        for i in range(1, m):
            Ti = U[i]
            delta = U[i] - U[i - 1]
            c[i] += -(S0 / A0) * delta * self.curve.discount(Ti)

        return U, c

    def _precompute_instruments(self):
        """
        Precompute, for each swaption i:
          - Tau, U, c
          - expiry T, DF(T)
          - A0, S0
          - K (rate units), N
          - market price
          - payer flag (if provided)
        """
        prices = self.market_prices["Prices"]
        strikes = self.market_prices["Strike"]
        notionals = self.market_prices["Notional"]
        dates = self.market_prices["Dates"]

        n = len(prices)
        instr = []

        for i in range(n):
            Tau = [float(x) for x in dates[i]]
            if len(Tau) < 2:
                raise ValueError(f"Instrument {i}: Tau must contain at least [T0, Tn].")

            T = float(Tau[0])
            DF = float(self.curve.discount(T))

            A0, S0 = self._annuity_and_swap_rate_0(Tau)
            U, c = self._frozen_weights(Tau, A0, S0)

            K = float(strikes[i]) / 100.0
            N = float(notionals[i])
            mkt = float(prices[i])

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
    # Outer-dependent Q's
    # -------------------------

    def _compute_Qs_for_outer(self, a, b):
        """
        For current outer (a,b), compute Qaa, Qbb, Qab for each instrument:
            Qaa = c^T I_aa c, etc.

        Uses the closed-form integrals implemented in HullWhite2FModel via the pricer/model.
        """
        Qaa = np.zeros(len(self._instr), dtype=float)
        Qbb = np.zeros(len(self._instr), dtype=float)
        Qab = np.zeros(len(self._instr), dtype=float)

        I_aa = self.model.__class__.I_aa if hasattr(self.model.__class__, "I_aa") else None
        I_bb = self.model.__class__.I_bb if hasattr(self.model.__class__, "I_bb") else None
        I_ab = self.model.__class__.I_ab if hasattr(self.model.__class__, "I_ab") else None

        if I_aa is None or I_bb is None or I_ab is None:
            from models.hw2f import HullWhite2FModel
            I_aa = HullWhite2FModel.I_aa
            I_bb = HullWhite2FModel.I_bb
            I_ab = HullWhite2FModel.I_ab

        for k, ins in enumerate(self._instr):
            T = ins["T"]
            U = ins["U"]
            c = ins["c"]

            qaa = 0.0
            qbb = 0.0
            qab = 0.0

            for i, Ui in enumerate(U):
                ci = c[i]
                if ci == 0.0:
                    continue
                for j, Uj in enumerate(U):
                    cj = c[j]
                    if cj == 0.0:
                        continue
                    qaa += ci * cj * I_aa(T, Ui, Uj, a)
                    qbb += ci * cj * I_bb(T, Ui, Uj, b)
                    qab += ci * cj * I_ab(T, Ui, Uj, a, b)

            Qaa[k] = qaa
            Qbb[k] = qbb
            Qab[k] = qab

        return Qaa, Qbb, Qab

    # -------------------------
    # Inner objective (sigma, eta)
    # -------------------------

    def _price_swaption_from_Qs(self, ins, rho, sigma, eta, qaa, qbb, qab):
        """
        Fast pricing using precomputed Q's:
          Var[S(T)] = sigma^2 qaa + eta^2 qbb + 2 rho sigma eta qab
          Bachelier on swap rate with annuity A0 and forward S0
        """
        T = ins["T"]
        A0 = ins["A0"]
        S0 = ins["S0"]
        K = ins["K"]
        N = ins["N"]
        payer = ins["payer"]

        varS = (sigma * sigma) * qaa + (eta * eta) * qbb + 2.0 * rho * sigma * eta * qab
        varS = float(max(varS, 0.0))

        w = 1.0 if payer else -1.0

        if T <= 0:
            return float(N * A0 * max(w * (S0 - K), 0.0))

        if varS < 1e-30:
            return float(N * A0 * max(w * (S0 - K), 0.0))

        sigmaN = np.sqrt(varS / T)
        d = (S0 - K) / (sigmaN * np.sqrt(T))

        price = N * A0 * (w * (S0 - K) * norm.cdf(w * d) + sigmaN * np.sqrt(T) * norm.pdf(d))
        return float(price)

    def _inner_objective(self, x, rho, Qaa, Qbb, Qab):
        """
        x = (log sigma, log eta). Returns RMSRE on prices or forward premiums.
        """
        sigma = float(np.exp(x[0]))
        eta = float(np.exp(x[1]))

        eps = 1e-6
        n = len(self._instr)

        err = 0.0
        for k, ins in enumerate(self._instr):
            model_pv = self._price_swaption_from_Qs(ins, rho, sigma, eta, Qaa[k], Qbb[k], Qab[k])

            if self.use_forward_premium:
                model_val = model_pv / (ins["DF"] + 1e-18)
            else:
                model_val = model_pv

            mkt = ins["mkt"]
            err += (1.0 / n) * ((model_val - mkt) ** 2) / (mkt * mkt + eps)

        rmsre = float(np.sqrt(err))
        self.inner_history.append((sigma, eta, rmsre))
        return rmsre

    def _inner_callback(self, x):
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
        Calibrate (sigma, eta) for fixed rho and precomputed Q's.
        """
        self.inner_history = []

        x0 = np.log([init_sigma, init_eta])
        bounds = [
            (np.log(bounds_sigma[0]), np.log(bounds_sigma[1])),
            (np.log(bounds_eta[0]), np.log(bounds_eta[1])),
        ]

        cb = self._inner_callback if verbose else None

        res = minimize(
            lambda x: self._inner_objective(x, rho, Qaa, Qbb, Qab),
            x0,
            bounds=bounds,
            method=method,
            callback=cb,
            options={"ftol": ftol},
        )

        sigma_opt = float(np.exp(res.x[0]))
        eta_opt = float(np.exp(res.x[1]))
        return res, sigma_opt, eta_opt

    # -------------------------
    # Public API: Profile calibration
    # -------------------------

    def calibrate_profile(
        self,
        # outer grid
        grid_a=None,
        grid_b=None,
        grid_rho=None,
        # inner init/bounds
        init_sigma=0.01,
        init_eta=0.008,
        bounds_sigma=(1e-4, 0.5),
        bounds_eta=(1e-4, 0.5),
        # optim settings
        inner_method="L-BFGS-B",
        inner_ftol=1e-6,
        verbose_inner=False,
        top_k=3,
    ):
        """
        Run profile calibration:
          - Evaluate a grid of (a,b,rho)
          - For each, compute Q's and calibrate (sigma,eta)
          - Keep best result

        Returns
        -------
        dict with best parameters and error.
        """
        if grid_a is None:
            grid_a = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30]
        if grid_b is None:
            grid_b = [0.001, 0.003, 0.01, 0.02, 0.05, 0.10]
        if grid_rho is None:
            grid_rho = [-0.95, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4]

        # ensure b < a (common identification)
        outer_candidates = []
        for a, b, rho in itertools.product(grid_a, grid_b, grid_rho):
            if b < a:
                outer_candidates.append((float(a), float(b), float(rho)))

        if not outer_candidates:
            raise ValueError("Empty outer grid after applying constraint b < a.")

        outer_total = len(outer_candidates)

        print(f"Profile calibration on {len(self._instr)} swaptions.")
        print(f"Outer grid candidates: {outer_total}")

        results = []
        best_rmsre = float("inf")

        for idx, (a, b, rho) in enumerate(outer_candidates, start=1):
            # --- progress: candidate start ---
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
                    # never break calibration due to UI callback
                    pass

            # set outer params (sigma,eta will be set by inner)
            print(f"\n[Outer {idx}/{outer_total}] a={a:.4f}, b={b:.4f}, rho={rho:+.2f}")

            # update model outer params before computing Q's
            self.model.parameters["a"] = a
            self.model.parameters["b"] = b
            self.model.parameters["rho"] = rho

            # compute Q's for this (a,b)
            Qaa, Qbb, Qab = self._compute_Qs_for_outer(a, b)

            # inner calibration on (sigma,eta)
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

            rmsre = float(res_in.fun)
            print(f"  -> inner best: sigma={sigma_opt:.6f}, eta={eta_opt:.6f}, RMSRE={rmsre:.5e}")

            improved = rmsre < best_rmsre
            if improved:
                best_rmsre = rmsre

            # --- progress: candidate done ---
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

        # sort by error
        results.sort(key=lambda d: d["rmsre"])
        best = results[0]

        # set final params in model
        self._set_params(best["a"], best["b"], best["rho"], best["sigma"], best["eta"])

        print("\n=== Profile calibration result (best) ===")
        print(f"Total Error (RMSRE): {best['rmsre']:>+8.3%}")
        print("Parameters:")
        print(f"  a    : {best['a']:.6f}")
        print(f"  b    : {best['b']:.6f}")
        print(f"  rho  : {best['rho']:+.6f}")
        print(f"  sigma: {best['sigma']:.6f}")
        print(f"  eta  : {best['eta']:.6f}")

        print("\nPer-instrument report:")

        # Recompute Qs for best (a,b)
        Qaa_best, Qbb_best, Qab_best = self._compute_Qs_for_outer(best["a"], best["b"])

        rho = best["rho"]
        sigma = best["sigma"]
        eta = best["eta"]

        for k, ins in enumerate(self._instr):
            model_pv = self._price_swaption_from_Qs(ins, rho, sigma, eta, Qaa_best[k], Qbb_best[k], Qab_best[k])

            if self.use_forward_premium:
                model_val = model_pv / (ins["DF"] + 1e-18)
            else:
                model_val = model_pv

            mkt = ins["mkt"]
            dif = model_val / (mkt + 1e-12) - 1.0

            Tau = ins["Tau"]
            print(
                f"Swaption {k:>3}: {Tau[0]:>5.2f}Y to {Tau[-1]:<5.2f}Y | "
                f"Model: {model_val:>10.4f} | Market: {mkt:>10.4f} | Diff: {dif:>+8.3%}"
            )

        # Print top_k candidates summary
        if top_k and top_k > 1:
            print(f"\nTop {min(top_k, len(results))} candidates:")
            for j in range(min(top_k, len(results))):
                r = results[j]
                print(
                    f"  {j+1:>2}. a={r['a']:.4f}, b={r['b']:.4f}, rho={r['rho']:+.2f} | "
                    f"sigma={r['sigma']:.5f}, eta={r['eta']:.5f} | RMSRE={r['rmsre']:.5e}"
                )

        # Return best + full ranking
        return {"best": best, "ranking": results}
