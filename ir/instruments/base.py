# -*- coding: utf-8 -*-
"""
ir/instruments/base.py

Couche "métier" :
- Instrument (base) : wrapper pour pricing (sans toucher aux formules)
- QuoteSet : encapsule un DataFrame et produit les dicts attendus par les calibrators
- SwaptionQuoteSet / CapletQuoteSet : implémentations concrètes

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
    """
    Instrument abstrait : expose uniquement une API commune.

    Idée :
    - chaque instrument concret (caplet, swaption, swap, etc.) héritera de cette classe
    - il devra implémenter price(pricer) et déléguer le calcul à un pricer existant
    - on standardise juste l’interface
    """
    def price(self, pricer: Any) -> float:
        # Méthode à implémenter par les classes filles
        raise NotImplementedError


# -------------------------
# QuoteSets (calibration / comparaison)
# -------------------------

@dataclass
class QuoteSet:
    """
    Classe de base : "wrappe" un DataFrame et définit une interface standard.

    Hypothèse :
    - df contient au minimum les colonnes nécessaires (définies par les attributs *_col)
    - les classes filles fourniront des méthodes utilitaires (to_market_dict, pricing, etc.)
    """
    df: pd.DataFrame

    def copy_df(self) -> pd.DataFrame:
        # Copie profonde : évite les effets de bord quand on ajoute des colonnes (Model_Price, erreurs, etc.)
        return self.df.copy(deep=True)


@dataclass
class SwaptionQuoteSet(QuoteSet):
    """
    Wrapper pour un template de calibration swaptions.

    Colonnes requises (par défaut) :
      - price_col, strike_col, notional_col, dates_col
    Colonne optionnelle :
      - payer_col (bool) : True=payer, False=receiver

    Objectif :
    - fournir le dict EXACT attendu par les calibrators
    - fournir des helpers de comparaison modèle vs marché (1F et 2F)
    """
    price_col: str = "Price"
    strike_col: str = "Strike"
    notional_col: str = "Notional"
    dates_col: str = "Payment_Dates"
    payer_col: Optional[str] = None

    def to_market_dict(self) -> dict:
        """
        Produit le dict EXACT attendu par les calibrators (1F et 2F profile) :
          - "Prices", "Strike", "Notional", "Dates" (+ éventuellement "Payer")

        Important :
        - Strike est gardé en % (ex: 3.0) car les calibrators font eux-mêmes /100.
        - Dates doit être une liste de listes (list[list[float]]).
        """
        # Conversion explicite en float pour éviter des types pandas/numpy inattendus
        d = {
            "Prices": self.df[self.price_col].astype(float).tolist(),
            "Strike": self.df[self.strike_col].astype(float).tolist(),
            "Notional": self.df[self.notional_col].astype(float).tolist(),
            "Dates": self.df[self.dates_col].tolist(),  # list[list[float]]
        }

        # Ajoute le flag payer si la colonne est fournie et existe
        if self.payer_col and self.payer_col in self.df.columns:
            d["Payer"] = self.df[self.payer_col].astype(bool).tolist()

        return d

    # ---- comparaison mkt vs modèle (pricing) ---- #

    def with_model_prices_1f(
        self,
        pricer: Any,
        forward_premium: bool = True,
        payer_default: bool = True
    ) -> pd.DataFrame:
        """
        Calcule les prix modèle pour toutes les lignes en utilisant un pricer HW1F.

        Convention :
        - Si forward_premium=True : retourne PV / DF(T0)
          (cohérent avec le notebook et la convention de calibration swaptions 1F)
        - Sinon : retourne PV directement

        Sortie :
        - Ajoute deux colonnes :
          * "Model_Price" : prix modèle selon la convention choisie
          * "Rel_Error"   : erreur relative (Model/Market - 1)
        """
        out = self.copy_df()
        model_prices = []

        # Boucle sur chaque swaption (ligne du DataFrame)
        for i in range(len(out)):
            # Dates du swap sous-jacent (T0, ..., Tn)
            Tau = out.loc[i, self.dates_col]

            # Notional
            N = float(out.loc[i, self.notional_col])

            # Strike (% -> taux)
            K = float(out.loc[i, self.strike_col]) / 100.0

            # Détermine payer/receiver : par défaut payer, sinon lecture depuis colonne
            payer = payer_default
            if self.payer_col and self.payer_col in out.columns:
                payer = bool(out.loc[i, self.payer_col])

            # PV modèle (swaption) via pricer 1F
            pv = float(pricer.swaption(Tau, N, K, payer=payer))

            # Conversion éventuelle PV -> prime forward (division par DF(T0))
            if forward_premium:
                T0 = float(Tau[0])
                df0 = float(pricer.curve.discount(T0))
                model_prices.append(pv / (df0 + 1e-18))
            else:
                model_prices.append(pv)

        # Ajout des colonnes de sortie
        out["Model_Price"] = model_prices

        # Erreur relative stabilisée (évite div par 0)
        out["Rel_Error"] = out["Model_Price"] / (out[self.price_col].astype(float) + 1e-12) - 1.0
        return out

    def with_model_prices_2f(
        self,
        pricer2f: Any,
        forward_premium: bool = True,
        payer_default: bool = True
    ) -> pd.DataFrame:
        """
        Calcule les prix modèle pour toutes les lignes avec un pricer HW2F (approx gaussienne).

        Méthode utilisée :
          pv = pricer2f.swaption_approx_hw2f(Tau, N, K, payer=?)

        Convention :
        - Si forward_premium=True : retourne PV / DF(T0)
        - Sinon : retourne PV

        Sortie :
        - Ajoute :
          * "Model_Price"
          * "Rel_Error"
        """
        out = self.copy_df()
        model_prices = []

        # Boucle sur les instruments
        for i in range(len(out)):
            Tau = out.loc[i, self.dates_col]
            N = float(out.loc[i, self.notional_col])
            K = float(out.loc[i, self.strike_col]) / 100.0

            payer = payer_default
            if self.payer_col and self.payer_col in out.columns:
                payer = bool(out.loc[i, self.payer_col])

            # PV modèle via approximation HW2F
            pv = float(pricer2f.swaption_approx_hw2f(Tau, N, K, payer=payer))

            # Conversion PV -> prime forward si demandé
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
    Wrapper pour un template de calibration caplets.

    Colonnes requises (par défaut) :
      - price_col, strike_col, notional_col, expiry_col, maturity_col

    Les calibrators attendent le dict :
      Prices, Strike, Notional, Expiry, Maturity
    """
    price_col: str = "Price"
    strike_col: str = "Strike"
    notional_col: str = "Notional"
    expiry_col: str = "Expiry"
    maturity_col: str = "Maturity"

    def to_market_dict(self) -> dict:
        """
        Produit le dict attendu par le calibrator 1F pour caplets.

        Important :
        - Strike est gardé en % car le calibrator fait /100 en interne.
        """
        return {
            "Prices": self.df[self.price_col].astype(float).tolist(),
            "Strike": self.df[self.strike_col].astype(float).tolist(),
            "Notional": self.df[self.notional_col].astype(float).tolist(),
            "Expiry": self.df[self.expiry_col].astype(float).tolist(),
            "Maturity": self.df[self.maturity_col].astype(float).tolist(),
        }

    def with_model_prices_1f(self, pricer: Any) -> pd.DataFrame:
        """
        Calcule les PV modèle des caplets via HW1F :
          PV = pricer.caplet(T1, T2, N, K)

        Sortie :
        - Ajoute :
          * "Model Price"
          * "Rel_Error"
        """
        out = self.copy_df()
        model_prices = []

        # Boucle sur les caplets (ligne par instrument)
        for i in range(len(out)):
            T1 = float(out.loc[i, self.expiry_col])
            T2 = float(out.loc[i, self.maturity_col])
            N = float(out.loc[i, self.notional_col])
            K = float(out.loc[i, self.strike_col]) / 100.0

            # Prix modèle caplet
            model_prices.append(float(pricer.caplet(T1, T2, N, K)))

        out["Model Price"] = model_prices
        out["Rel_Error"] = out["Model Price"] / (out[self.price_col].astype(float) + 1e-12) - 1.0
        return out

    def with_model_prices_2f(self, pricer2f: Any) -> pd.DataFrame:
        """
        Calcule les PV modèle des caplets via HW2F :
          PV = pricer2f.caplet_hw2f(T1, T2, N, K)

        Sortie :
        - Ajoute :
          * "Model Price"
          * "Rel_Error"
        """
        out = self.copy_df()
        model_prices = []

        for i in range(len(out)):
            T1 = float(out.loc[i, self.expiry_col])
            T2 = float(out.loc[i, self.maturity_col])
            N = float(out.loc[i, self.notional_col])
            K = float(out.loc[i, self.strike_col]) / 100.0

            # Prix modèle caplet 2F
            model_prices.append(float(pricer2f.caplet_hw2f(T1, T2, N, K)))

        out["Model Price"] = model_prices
        out["Rel_Error"] = out["Model Price"] / (out[self.price_col].astype(float) + 1e-12) - 1.0
        return out


# -------------------------
# util simple
# -------------------------

def worst_rows_by_abs_relerr(df: pd.DataFrame, relerr_col: str = "Rel_Error", n: int = 10) -> pd.DataFrame:
    """
    Renvoie les n lignes avec les plus grosses erreurs relatives en valeur absolue.

    Utile pour diagnostiquer une calibration/pricing :
    - on repère rapidement les instruments les plus mal "fit"
    - on peut ensuite inspecter leurs caractéristiques (maturité, strike, etc.)

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame contenant au moins la colonne relerr_col.
    relerr_col : str
        Nom de la colonne d’erreur relative (par défaut "Rel_Error").
    n : int
        Nombre de lignes à renvoyer.

    Retourne
    --------
    pd.DataFrame
        Sous-ensemble trié par erreur absolue décroissante.
    """
    out = df.copy()
    out["AbsRelErr"] = np.abs(out[relerr_col].astype(float))
    return out.sort_values("AbsRelErr", ascending=False).head(int(n))
