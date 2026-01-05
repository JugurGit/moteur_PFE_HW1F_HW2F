# -*- coding: utf-8 -*-
"""
ir/market/loaders_excel.py

Loaders Excel -> objets "propres" (Curve + QuoteSets) pour notebooks.

Objectifs:
- Remplacer le glue-code notebook:
    pd.read_excel(...), parse Payment_Dates, construire market_prices dict
- Sans changer tes pricers/calibrators: on retourne exactement ce qu'ils attendent.

Usage typique notebook:
    curve = load_curve_xlsx(path)
    swpn = load_swaption_template_xlsx(path)
    market_dict = swpn.to_market_dict()
"""

from __future__ import annotations

import ast
from typing import Optional

import pandas as pd

# Imports robustes (migration progressive)
try:
    from ir.market.curve import Curve
except Exception:  # pragma: no cover
    from curve_builder import Curve  # type: ignore

try:
    from ir.instruments.base import SwaptionQuoteSet, CapletQuoteSet
except Exception:  # pragma: no cover
    from instruments.base import SwaptionQuoteSet, CapletQuoteSet  # type: ignore


def _parse_list_cell(x) -> list[float]:
    """
    Parse une cellule Excel supposée contenir une liste:
      - déjà list/tuple -> cast float
      - string "[0.5, 1.0, ...]" -> ast.literal_eval
    """
    if isinstance(x, (list, tuple)):
        return [float(v) for v in x]
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            obj = ast.literal_eval(s)  # safe eval (contrairement à eval)
            if isinstance(obj, (list, tuple)):
                return [float(v) for v in obj]
        except Exception:
            pass
    # fallback
    raise ValueError(f"Cannot parse payment dates cell: {x!r}")


def load_curve_xlsx(
    path: str,
    sheet: str = "Curve",
    time_col: str = "Year_Frac",
    df_col: str = "Discount_Factor",
    smooth: float = 1e-7,
) -> Curve:
    """
    Read curve sheet and build Curve(time, discount_factors).

    Parameters
    ----------
    path: xlsx path
    sheet: sheet name containing curve nodes
    time_col: column with year fractions
    df_col: column with discount factors
    smooth: passed to Curve (forward spline smoothing)
    """
    df = pd.read_excel(path, sheet_name=sheet)
    time = df[time_col].astype(float).values
    disc = df[df_col].astype(float).values
    return Curve(time, disc, smooth=smooth)


def load_swaption_template_xlsx(
    path: str,
    sheet: str = "Template",
    payment_dates_col: str = "Payment_Dates",
    price_col: str = "Price",
    strike_col: str = "Strike",
    notional_col: str = "Notional",
    payer_col: Optional[str] = None,
) -> SwaptionQuoteSet:
    """
    Read swaption calibration template (your SWPN_Calibration_Template_...xlsx).

    Expected columns (minimum):
      - Price, Strike, Notional, Payment_Dates
    Optionally:
      - Payer (bool) if provided in payer_col or column named "Payer"
    """
    df = pd.read_excel(path, sheet_name=sheet)

    # Payment_Dates: string like "[0.5, 1.0, ...]" -> list[float]
    if payment_dates_col in df.columns:
        df[payment_dates_col] = [_parse_list_cell(v) for v in df[payment_dates_col].tolist()]
    else:
        raise KeyError(f"Missing column '{payment_dates_col}' in {sheet}.")

    # Payer flags (optional)
    if payer_col and payer_col in df.columns:
        df["Payer"] = df[payer_col].astype(bool)
    elif "Payer" in df.columns:
        df["Payer"] = df["Payer"].astype(bool)

    return SwaptionQuoteSet(
        df=df,
        price_col=price_col,
        strike_col=strike_col,
        notional_col=notional_col,
        dates_col=payment_dates_col,
        payer_col=("Payer" if "Payer" in df.columns else None),
    )


def load_caplet_template_xlsx(
    path: str,
    sheet: str = "Template",
    drop_first_row_if_empty: bool = True,
    price_col: str = "Price",
    strike_col: str = "Strike",
    notional_col: str = "Notional",
    expiry_col: str = "Expiry",
    maturity_col: str = "Maturity",
) -> CapletQuoteSet:
    """
    Read caplet calibration template (your CAP_Calibration_Template_...xlsx).

    Notes
    -----
    In your notebook you did: df = pd.read_excel(...).iloc[1:,:]
    Sometimes first row is blank/header-like; we offer drop_first_row_if_empty.
    """
    df = pd.read_excel(path, sheet_name=sheet)

    if drop_first_row_if_empty and len(df) >= 1:
        # heuristic: if first row has NaNs for essential fields -> drop it
        essentials = [price_col, strike_col, notional_col, expiry_col, maturity_col]
        if any(col in df.columns for col in essentials):
            row0 = df.iloc[0]
            bad = True
            for col in essentials:
                if col in df.columns and pd.notna(row0[col]):
                    bad = False
                    break
            if bad:
                df = df.iloc[1:, :].reset_index(drop=True)

    return CapletQuoteSet(
        df=df.reset_index(drop=True),
        price_col=price_col,
        strike_col=strike_col,
        notional_col=notional_col,
        expiry_col=expiry_col,
        maturity_col=maturity_col,
    )


def load_cap_market_data_xlsx(path: str) -> pd.DataFrame:
    """
    Read OTM caplet market data file (your CAP_Market_Data_...xlsx).
    Returns DataFrame as-is (smile filtering remains notebook-side, by design).
    """
    return pd.read_excel(path)
