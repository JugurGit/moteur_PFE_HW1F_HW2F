# -*- coding: utf-8 -*-
"""
ir/instruments/base.py

Couche "métier" légère:
- Instrument (base) : wrapper pour pricing (sans toucher aux formules)
- QuoteSet : encapsule un DataFrame et sait produire les dicts attendus par tes calibrators
- SwaptionQuoteSet / CapletQuoteSet : implémentations concrètes

But:
- réduire drastiquement le glue-code notebook
- garder tes calibrators/pricers inchangés
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any

import numpy as np
import pandas as pd


# -------------------------
# Instruments (base)
# -------------------------

@dataclass(frozen=True)
class Instrument:
    """Instrument abstrait: seulement une API commune."""
    def price(self, pricer: Any) -> float:
        raise NotImplementedError


# -------------------------
# QuoteSets (calibration / comparaison)
# -------------------------

@dataclass
class QuoteSet:
    """
    Base class: wraps a DataFrame and defines a standard interface.

    df is expected to contain at least the columns referenced by *_col attributes.
    """
    df: pd.DataFrame

    def copy_df(self) -> pd.DataFrame:
        return self.df.copy(deep=True)


@dataclass
class SwaptionQuoteSet(QuoteSet):
    """
    Wrap swaption calibration template.

    Required columns:
      - price_col, strike_col, notional_col, dates_col
    Optional:
      - payer_col (bool)
    """
    price_col: str = "Price"
    strike_col: str = "Strike"
    notional_col: str = "Notional"
    dates_col: str = "Payment_Dates"
    payer_col: Optional[str] = None

    def to_market_dict(self) -> dict:
        """
        Dict EXACT attendu par tes calibrators (1F & 2F profile):
          Prices, Strike, Notional, Dates (+ optional Payer)
        Strike is kept in % (calibrator divides by 100 itself for 1F and 2F profile).
        """
        d = {
            "Prices": self.df[self.price_col].astype(float).tolist(),
            "Strike": self.df[self.strike_col].astype(float).tolist(),
            "Notional": self.df[self.notional_col].astype(float).tolist(),
            "Dates": self.df[self.dates_col].tolist(),  # list[list[float]]
        }
        if self.payer_col and self.payer_col in self.df.columns:
            d["Payer"] = self.df[self.payer_col].astype(bool).tolist()
        return d

    # ---- comparaison mkt vs modèle (pricing) ---- #

    def with_model_prices_1f(self, pricer: Any, forward_premium: bool = True, payer_default: bool = True) -> pd.DataFrame:
        """
        Compute model prices for ALL rows using HW1F pricer.

        - If forward_premium=True: return PV/DF(T0) (matches your notebook & calibrator convention)
        - Else: return PV
        """
        out = self.copy_df()
        model_prices = []

        for i in range(len(out)):
            Tau = out.loc[i, self.dates_col]
            N = float(out.loc[i, self.notional_col])
            K = float(out.loc[i, self.strike_col]) / 100.0
            payer = payer_default
            if self.payer_col and self.payer_col in out.columns:
                payer = bool(out.loc[i, self.payer_col])

            pv = float(pricer.swaption(Tau, N, K, payer=payer))
            if forward_premium:
                T0 = float(Tau[0])
                df0 = float(pricer.curve.discount(T0))
                model_prices.append(pv / (df0 + 1e-18))
            else:
                model_prices.append(pv)

        out["Model_Price"] = model_prices
        out["Rel_Error"] = out["Model_Price"] / (out[self.price_col].astype(float) + 1e-12) - 1.0
        return out

    def with_model_prices_2f(self, pricer2f: Any, forward_premium: bool = True, payer_default: bool = True) -> pd.DataFrame:
        """
        Compute model prices for ALL rows using HW2F pricer (Gaussian approx).

        Uses:
          pv = pricer2f.swaption_approx_hw2f(Tau, N, K, payer=?)
        and optionally converts to forward premium dividing by DF(T0).
        """
        out = self.copy_df()
        model_prices = []

        for i in range(len(out)):
            Tau = out.loc[i, self.dates_col]
            N = float(out.loc[i, self.notional_col])
            K = float(out.loc[i, self.strike_col]) / 100.0
            payer = payer_default
            if self.payer_col and self.payer_col in out.columns:
                payer = bool(out.loc[i, self.payer_col])

            pv = float(pricer2f.swaption_approx_hw2f(Tau, N, K, payer=payer))
            if forward_premium:
                T0 = float(Tau[0])
                df0 = float(pricer2f.curve.discount(T0))
                model_prices.append(pv / (df0 + 1e-18))
            else:
                model_prices.append(pv)

        out["Model_Price"] = model_prices
        out["Rel_Error"] = out["Model_Price"] / (out[self.price_col].astype(float) + 1e-12) - 1.0
        return out


@dataclass
class CapletQuoteSet(QuoteSet):
    """
    Wrap caplet calibration template.

    Required columns:
      - price_col, strike_col, notional_col, expiry_col, maturity_col

    Your calibrator expects:
      Prices, Strike, Notional, Expiry, Maturity
    """
    price_col: str = "Price"
    strike_col: str = "Strike"
    notional_col: str = "Notional"
    expiry_col: str = "Expiry"
    maturity_col: str = "Maturity"

    def to_market_dict(self) -> dict:
        return {
            "Prices": self.df[self.price_col].astype(float).tolist(),
            "Strike": self.df[self.strike_col].astype(float).tolist(),
            "Notional": self.df[self.notional_col].astype(float).tolist(),
            "Expiry": self.df[self.expiry_col].astype(float).tolist(),
            "Maturity": self.df[self.maturity_col].astype(float).tolist(),
        }

    def with_model_prices_1f(self, pricer: Any) -> pd.DataFrame:
        """
        Compute model PVs for caplets using HW1F pricer.caplet(T1,T2,N,K).
        """
        out = self.copy_df()
        model_prices = []

        for i in range(len(out)):
            T1 = float(out.loc[i, self.expiry_col])
            T2 = float(out.loc[i, self.maturity_col])
            N = float(out.loc[i, self.notional_col])
            K = float(out.loc[i, self.strike_col]) / 100.0
            model_prices.append(float(pricer.caplet(T1, T2, N, K)))

        out["Model Price"] = model_prices
        out["Rel_Error"] = out["Model Price"] / (out[self.price_col].astype(float) + 1e-12) - 1.0
        return out

    def with_model_prices_2f(self, pricer2f: Any) -> pd.DataFrame:
        """
        Compute model PVs for caplets using HW2F pricer.caplet_hw2f(T1,T2,N,K).
        """
        out = self.copy_df()
        model_prices = []

        for i in range(len(out)):
            T1 = float(out.loc[i, self.expiry_col])
            T2 = float(out.loc[i, self.maturity_col])
            N = float(out.loc[i, self.notional_col])
            K = float(out.loc[i, self.strike_col]) / 100.0
            model_prices.append(float(pricer2f.caplet_hw2f(T1, T2, N, K)))

        out["Model Price"] = model_prices
        out["Rel_Error"] = out["Model Price"] / (out[self.price_col].astype(float) + 1e-12) - 1.0
        return out


# -------------------------
# util simple
# -------------------------

def worst_rows_by_abs_relerr(df: pd.DataFrame, relerr_col: str = "Rel_Error", n: int = 10) -> pd.DataFrame:
    out = df.copy()
    out["AbsRelErr"] = np.abs(out[relerr_col].astype(float))
    return out.sort_values("AbsRelErr", ascending=False).head(int(n))
