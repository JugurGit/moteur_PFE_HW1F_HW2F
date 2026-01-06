# -*- coding: utf-8 -*-
"""
ir/instruments/rates.py

Instruments IR : wrappers "POO" qui délèguent aux pricers existants.

Idée :
- uniformiser le pricing (instrument.price(pricer))
- construire des portfolios (list[Instrument])

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ir.instruments.base import Instrument


@dataclass(frozen=True)
class Caplet(Instrument):
    """
    Caplet (un seul coupon de cap) défini par :
    - période [T1, T2]
    - nominal N
    - strike K (en taux, ex 0.03)

    Objectif :
    - fournir une interface unique Instrument.price(pricer)
    - déléguer au pricer 1F ou 2F selon ce qui est disponible
    """
    T1: float
    T2: float
    N: float
    K: float  # en taux (0.03)

    def price(self, pricer: Any) -> float:
        # Si le pricer expose une méthode 2F dédiée, on l'utilise
        # Sinon, fallback sur la méthode 1F.
        if hasattr(pricer, "caplet_hw2f"):
            return float(pricer.caplet_hw2f(self.T1, self.T2, self.N, self.K))
        return float(pricer.caplet(self.T1, self.T2, self.N, self.K))


@dataclass(frozen=True)
class Floorlet(Instrument):
    """
    Floorlet (un seul coupon de floor) défini par :
    - période [T1, T2]
    - nominal N
    - strike K

    """
    T1: float
    T2: float
    N: float
    K: float

    def price(self, pricer: Any) -> float:
        # Si une méthode 2F explicite existe, on la privilégie
        if hasattr(pricer, "floorlet_hw2f"):
            return float(pricer.floorlet_hw2f(self.T1, self.T2, self.N, self.K))

        # Sinon, on appelle le pricer.floor sur 2 dates pour représenter un seul coupon
        Tau = [float(self.T1), float(self.T2)]
        return float(pricer.floor(Tau, self.N, self.K))


@dataclass(frozen=True)
class Cap(Instrument):
    """
    Cap (ensemble de caplets) défini par :
    - Tau : séquence de dates [T0, T1, ..., Tn]
    - nominal N
    - strike K

    Remarque :
    - Délègue à pricer.cap (1F) ou pricer.cap_hw2f (2F) si disponible.
    """
    Tau: Sequence[float]
    N: float
    K: float

    def price(self, pricer: Any) -> float:
        # Si le pricer 2F existe, on l'utilise
        if hasattr(pricer, "cap_hw2f"):
            return float(pricer.cap_hw2f(list(self.Tau), self.N, self.K))
        # Sinon 1F
        return float(pricer.cap(list(self.Tau), self.N, self.K))


@dataclass(frozen=True)
class Floor(Instrument):
    """
    Floor (ensemble de floorlets) défini par :
    - Tau : séquence de dates [T0, T1, ..., Tn]
    - nominal N
    - strike K

    Remarque :
    - Délègue à pricer.floor (1F) ou pricer.floor_hw2f (2F) si disponible.
    """
    Tau: Sequence[float]
    N: float
    K: float

    def price(self, pricer: Any) -> float:
        # Si le pricer 2F existe, on l'utilise
        if hasattr(pricer, "floor_hw2f"):
            return float(pricer.floor_hw2f(list(self.Tau), self.N, self.K))
        # Sinon 1F
        return float(pricer.floor(list(self.Tau), self.N, self.K))


@dataclass(frozen=True)
class Swap(Instrument):
    """
    Swap vanilla (jambe fixe vs flottante) défini par :
    - Tau : séquence de dates [T0, T1, ..., Tn] (échéances de paiement)
    - nominal N
    - taux fixe K
    - payer : True si payer fixe / receive float, False sinon

    Note :
    - Ici, on suppose que pricer.swap est disponible et gère payer/receiver.
    """
    Tau: Sequence[float]
    N: float
    K: float
    payer: bool = True

    def price(self, pricer: Any) -> float:
        # Délégation directe : aucun calcul ici
        return float(pricer.swap(list(self.Tau), self.N, self.K, payer=self.payer))


@dataclass(frozen=True)
class Swaption(Instrument):
    """
    Swaption (option sur swap) définie par :
    - Tau : dates du swap sous-jacent [T0, T1, ..., Tn]
    - nominal N
    - strike K
    - payer : True (payer swaption) ou False (receiver swaption)
    - forward_premium : si True, retourne PV / DF(T0) pour coller à la convention de calibration

    Remarque :
    - Si un pricer 2F approx est disponible (swaption_approx_hw2f), on l'utilise.
    - Sinon, fallback sur pricer.swaption (1F).
    """
    Tau: Sequence[float]
    N: float
    K: float
    payer: bool = True
    forward_premium: bool = True  # pour matcher le notebook/calibration

    def price(self, pricer: Any) -> float:
        Tau = list(self.Tau)

        # --- Cas 2F : pricer expose une approximation gaussienne dédiée ---
        if hasattr(pricer, "swaption_approx_hw2f"):
            # PV 2F approx
            pv = float(pricer.swaption_approx_hw2f(Tau, self.N, self.K, payer=self.payer))

            # Optionnel : conversion PV -> prime forward (division par DF(T0))
            if self.forward_premium:
                df0 = float(pricer.curve.discount(float(Tau[0])))
                return pv / (df0 + 1e-18)
            return pv

        # --- Cas 1F : méthode swaption "classique" ---
        pv = float(pricer.swaption(Tau, self.N, self.K, payer=self.payer))

        # Optionnel : conversion PV -> prime forward (même convention que ci-dessus)
        if self.forward_premium:
            df0 = float(pricer.curve.discount(float(Tau[0])))
            return pv / (df0 + 1e-18)
        return pv
