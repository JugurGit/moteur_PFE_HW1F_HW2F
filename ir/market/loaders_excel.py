# -*- coding: utf-8 -*-
"""
ir/market/loaders_excel.py

Loaders Excel -> objets "propres" (Curve + QuoteSets) pour notebooks.

Usage typique notebook :
    curve = load_curve_xlsx(path)
    swpn = load_swaption_template_xlsx(path)
    market_dict = swpn.to_market_dict()
"""

from __future__ import annotations

import ast
from typing import Optional
import pandas as pd
from ir.market.curve import Curve
from ir.instruments.base import SwaptionQuoteSet, CapletQuoteSet



def _parse_list_cell(x) -> list[float]:
    """
    Parse une cellule Excel supposée contenir une liste de dates (year fractions).

    Retourne
    --------
    list[float]

    Erreurs
    -------
    - ValueError si la cellule ne peut pas être interprétée comme une liste de floats.
    """
    # Cas 1 : déjà une liste/tuple (certains exports Excel ou pré-traitements peuvent faire ça)
    if isinstance(x, (list, tuple)):
        return [float(v) for v in x]

    # Cas 2 : string contenant une représentation de liste
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, (list, tuple)):
                return [float(v) for v in obj]
        except Exception:
            pass

    # Fallback : format non reconnu
    raise ValueError(f"Format de dates de paiement non reconnu: {x!r}")


def load_curve_xlsx(
    path: str,
    sheet: str = "Curve",
    time_col: str = "Year_Frac",
    df_col: str = "Discount_Factor",
    smooth: float = 1e-7,
) -> Curve:
    """
    Lit une feuille Excel contenant les nœuds de courbe et construit un objet Curve.

    Paramètres
    ----------
    path : str
        Chemin du fichier Excel (.xlsx).
    sheet : str
        Nom de la feuille contenant la courbe (par défaut "Curve").
    time_col : str
        Colonne contenant les maturités en années (year fractions).
    df_col : str
        Colonne contenant les discount factors P(0,T).
    smooth : float
        Paramètre de lissage transmis à Curve (utilisé pour la spline des forwards instantanés).

    Retourne
    --------
    Curve
        Objet courbe (interpolation DF + spline forward).
    """
    # Lecture Excel (pandas gère l'engine automatiquement dans la plupart des cas)
    df = pd.read_excel(path, sheet_name=sheet)

    # Extraction + cast explicite en float (sécurité type)
    time = df[time_col].astype(float).values
    disc = df[df_col].astype(float).values

    # Construction de la courbe
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
    Lit un fichier template de calibration swaptions (type SWPN_Calibration_Template_...xlsx)
    et retourne un SwaptionQuoteSet prêt à l’emploi.

    Colonnes attendues (minimum)
    ----------------------------
    - Price, Strike, Notional, Payment_Dates

    Optionnel
    ---------
    - Payer (bool) :
        * soit fourni via le paramètre payer_col (nom de colonne)
        * soit déjà présent dans le fichier sous le nom "Payer"

    """
    # Lecture de la feuille template
    df = pd.read_excel(path, sheet_name=sheet)

    if payment_dates_col in df.columns:
        df[payment_dates_col] = [_parse_list_cell(v) for v in df[payment_dates_col].tolist()]
    else:
        raise KeyError(f"Missing column '{payment_dates_col}' in {sheet}.")

    # --- Gestion du flag payer/receiver (optionnel) ---
    # Objectif : exposer une colonne standard "Payer" si possible
    if payer_col and payer_col in df.columns:
        # Si l'utilisateur indique une colonne, on la copie/normalise
        df["Payer"] = df[payer_col].astype(bool)
    elif "Payer" in df.columns:
        # Sinon, si la colonne s'appelle déjà "Payer", on la cast en bool
        df["Payer"] = df["Payer"].astype(bool)

    # Construction du QuoteSet avec la référence de colonnes
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
    Lit un fichier template de calibration caplets (type CAP_Calibration_Template_...xlsx)
    et retourne un CapletQuoteSet prêt à l’emploi.

    """
    # Lecture du template
    df = pd.read_excel(path, sheet_name=sheet)

    # Option : suppression heuristique de la première ligne si elle est "vide" sur les champs essentiels
    if drop_first_row_if_empty and len(df) >= 1:
        essentials = [price_col, strike_col, notional_col, expiry_col, maturity_col]
        # Petite protection : on ne lance l'heuristique que si au moins une colonne essentielle existe
        if any(col in df.columns for col in essentials):
            row0 = df.iloc[0]
            # On considère la ligne 0 "mauvaise" si toutes les colonnes essentielles sont NaN
            bad = True
            for col in essentials:
                if col in df.columns and pd.notna(row0[col]):
                    bad = False
                    break
            # Si la ligne 0 semble inutile, on la drop et on réindexe proprement
            if bad:
                df = df.iloc[1:, :].reset_index(drop=True)

    # Construction du QuoteSet caplets avec les colonnes configurées
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
    Lit un fichier de données de marché caplets OTM (type CAP_Market_Data_...xlsx)
    et renvoie le DataFrame tel quel.
    """
    return pd.read_excel(path)
