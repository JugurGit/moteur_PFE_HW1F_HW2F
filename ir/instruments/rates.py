# -*- coding: utf-8 -*-
"""
ir/instruments/rates.py

Instruments IR: wrappers "POO" qui délèguent à tes pricers existants.
Aucune formule n'est recodée ici.

Tu peux t'en servir pour:
- uniformiser le pricing (instrument.price(pricer))
- construire des portfolios (list[Instrument]) plus tard

Mais ton calibrage peut rester basé sur QuoteSets (base.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ir.instruments.base import Instrument


@dataclass(frozen=True)
class Caplet(Instrument):
    T1: float
    T2: float
    N: float
    K: float  # rate units (0.03)

    def price(self, pricer: Any) -> float:
        # HW1F: pricer.caplet ; HW2F: pricer.caplet_hw2f
        if hasattr(pricer, "caplet_hw2f"):
            return float(pricer.caplet_hw2f(self.T1, self.T2, self.N, self.K))
        return float(pricer.caplet(self.T1, self.T2, self.N, self.K))


@dataclass(frozen=True)
class Floorlet(Instrument):
    # pas de méthode dédiée en 2F dans ton code, donc on appelle 1F floorlet via put/call ZC
    # si tu ajoutes un floorlet_hw2f plus tard, cette classe le détectera.
    T1: float
    T2: float
    N: float
    K: float

    def price(self, pricer: Any) -> float:
        if hasattr(pricer, "floorlet_hw2f"):
            return float(pricer.floorlet_hw2f(self.T1, self.T2, self.N, self.K))
        # 1F: tu n'as pas de méthode floorlet explicite, donc on peut approx via floor([T1,T2],...)
        # mais pour rester strict "pas d'ajout de formule", on utilise floor sur 2 dates:
        Tau = [float(self.T1), float(self.T2)]
        return float(pricer.floor(Tau, self.N, self.K))


@dataclass(frozen=True)
class Cap(Instrument):
    Tau: Sequence[float]
    N: float
    K: float

    def price(self, pricer: Any) -> float:
        if hasattr(pricer, "cap_hw2f"):
            return float(pricer.cap_hw2f(list(self.Tau), self.N, self.K))
        return float(pricer.cap(list(self.Tau), self.N, self.K))


@dataclass(frozen=True)
class Floor(Instrument):
    Tau: Sequence[float]
    N: float
    K: float

    def price(self, pricer: Any) -> float:
        if hasattr(pricer, "floor_hw2f"):
            return float(pricer.floor_hw2f(list(self.Tau), self.N, self.K))
        return float(pricer.floor(list(self.Tau), self.N, self.K))


@dataclass(frozen=True)
class Swap(Instrument):
    Tau: Sequence[float]
    N: float
    K: float
    payer: bool = True

    def price(self, pricer: Any) -> float:
        return float(pricer.swap(list(self.Tau), self.N, self.K, payer=self.payer))


@dataclass(frozen=True)
class Swaption(Instrument):
    Tau: Sequence[float]
    N: float
    K: float
    payer: bool = True
    forward_premium: bool = True  # pour matcher ton notebook/calibration

    def price(self, pricer: Any) -> float:
        Tau = list(self.Tau)
        if hasattr(pricer, "swaption_approx_hw2f"):
            pv = float(pricer.swaption_approx_hw2f(Tau, self.N, self.K, payer=self.payer))
            if self.forward_premium:
                df0 = float(pricer.curve.discount(float(Tau[0])))
                return pv / (df0 + 1e-18)
            return pv

        pv = float(pricer.swaption(Tau, self.N, self.K, payer=self.payer))
        if self.forward_premium:
            df0 = float(pricer.curve.discount(float(Tau[0])))
            return pv / (df0 + 1e-18)
        return pv
