from scipy.optimize import minimize, brentq
import numpy as np
from scipy.stats import norm
import itertools
from typing import Callable, Optional, Dict, Any


class HullWhiteCalibrator:
    """
    Calibrates Hull–White 1F parameters (a, sigma) using market prices of Caplets or Swaptions.

    Notes
    -----
    - Uses log-parameterization to enforce positivity: a = exp(x[0]), sigma = exp(x[1]).
    - Objective: RMSRE (root mean squared relative error) on prices (or forward premiums for swaptions).
    - Keeps detailed prints (callback + final per-instrument report) like your original code.

    Patch (Streamlit progress)
    --------------------------
    - progress_cb: callable optional called at each optimizer iteration (via callback).
      It receives a dict: {"iter": int, "a": float, "sigma": float, "rmsre": float}
    """

    def __init__(
        self,
        pricer,
        market_prices,
        calibrate_to="Caplets",
        progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,  # NEW
    ):
        self.pricer = pricer
        self.model = pricer.model
        self.market_prices = market_prices
        self.calibrate_to = calibrate_to
        self.history = []  # stores tuples (a, sigma, rmsre)

        self.progress_cb = progress_cb  # NEW
        self._cb_iter = 0  # NEW: true optimizer-iteration counter (callback count)

    # -------- internal helpers -------- #

    def _set_params(self, a: float, sigma: float) -> None:
        self.model.parameters["a"] = float(a)
        self.model.parameters["sigma"] = float(sigma)

    def _price_instrument(self, i: int) -> float:
        """
        Returns model price (or forward premium for swaptions) for instrument i,
        using current model parameters (a, sigma).
        """
        K = self.market_prices["Strike"][i] / 100.0
        N = self.market_prices["Notional"][i]

        if self.calibrate_to == "Caplets":
            T = self.market_prices["Expiry"][i]
            S = self.market_prices["Maturity"][i]
            return self.pricer.caplet(T, S, N, K)

        elif self.calibrate_to == "Swaptions":
            Tau = self.market_prices["Dates"][i]
            DF = self.pricer.curve.discount(Tau[0])
            # Forward premium consistency: ensure market_prices['Prices'] matches this convention.
            return self.pricer.swaption(Tau, N, K) / DF

        raise ValueError("Calibration only implemented for 'Caplets' and 'Swaptions'.")

    # -------- optimization objective + callback -------- #

    def objective(self, x):
        """
        Objective function J(x) where x = (log(a), log(sigma)).
        Returns RMSRE over instruments.

        RMSRE = sqrt( (1/n) * sum_i ((model_i - mkt_i)^2 / (mkt_i^2 + eps)) )
        """
        a = np.exp(x[0])
        sigma = np.exp(x[1])
        self._set_params(a, sigma)

        prices = self.market_prices["Prices"]
        n = len(prices)
        eps = 1e-6

        err = 0.0
        for i in range(n):
            market_price = prices[i]
            model_price = self._price_instrument(i)
            err += (1.0 / n) * ((model_price - market_price) ** 2) / (market_price**2 + eps)

        rmsre = float(np.sqrt(err))
        self.history.append((a, sigma, rmsre))
        return rmsre

    def callback(self, x):
        """
        Print current (a, sigma) and error during optimization (like your original callback).
        Also calls progress_cb if provided (for Streamlit live UI).
        """
        if not self.history:
            return

        self._cb_iter += 1  # NEW
        a, sigma, err = self.history[-1]

        # keep your console print (capturable by your capture_stdout)
        print(f"a: {a:.6f}, sigma: {sigma:.6f}, RMSRE: {err:.5e}")

        # NEW: Streamlit/UI hook (safe)
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
                # Do NOT break optimization if UI update fails
                pass

    # -------- main entry point -------- #

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
        Run optimization to calibrate both a and sigma.

        Returns
        -------
        scipy.optimize.OptimizeResult
        """
        # reset callback iteration counter for each run
        self._cb_iter = 0

        # log-space init + bounds
        x0 = np.log([init_a, init_sigma])
        bounds = [
            (np.log(bounds_a[0]), np.log(bounds_a[1])),
            (np.log(bounds_sigma[0]), np.log(bounds_sigma[1])),
        ]

        result = minimize(
            self.objective,
            x0,
            bounds=bounds,
            method=method,
            callback=self.callback,
            options={"ftol": ftol},
        )

        if result.success:
            a_opt = float(np.exp(result.x[0]))
            sigma_opt = float(np.exp(result.x[1]))
            self._set_params(a_opt, sigma_opt)

            print("\nCalibration successful:")
            print(f"Iterations: {result.nit}")
            print(f"Number of instruments: {len(self.market_prices['Prices'])}")
            print(f"Total Error (RMSRE): {result.fun:>+8.3%}\n")
            print("Parameters:")
            print(f"Optimal a: {a_opt:.6f}")
            print(f"Optimal sigma: {sigma_opt:.6f}\n")

            # Per-instrument report
            for i in range(len(self.market_prices["Prices"])):
                market_price = self.market_prices["Prices"][i]
                model_price = self._price_instrument(i)
                dif = model_price / (market_price + 1e-12) - 1.0

                if self.calibrate_to == "Caplets":
                    T = self.market_prices["Expiry"][i]
                    S = self.market_prices["Maturity"][i]
                    print(
                        f"Caplet {i:>2}: {T:>5.2f}Y to {S:<5.2f}Y | "
                        f"Model: {model_price:>8.2f} | Market: {market_price:>8.2f} | Diff: {dif:>+8.3%}"
                    )

                elif self.calibrate_to == "Swaptions":
                    Tau = self.market_prices["Dates"][i]
                    print(
                        f"Swaption {i:>3}: {Tau[0]:>5.2f}Y to {Tau[-1]:<5.2f}Y | "
                        f"Model: {model_price:>8.2f} | Market: {market_price:>8.2f} | Diff: {dif:>+8.3%}"
                    )

        else:
            print("Calibration failed:", result.message)

        return result
