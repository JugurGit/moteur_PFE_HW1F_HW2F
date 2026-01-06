from scipy.optimize import minimize, brentq
import numpy as np
from scipy.stats import norm
import itertools


# ----------------------------
# Vol implicite Bachelier (normal) à partir d'un prix
# ----------------------------
def black_normal_vol(price, forward, strike, expiry, notional, annuity):
    """
    Calcule la volatilité implicite normale (Bachelier) à partir d'un prix de marché
    (swaption ou caplet).

    Conventions
    ----------
    - forward et strike sont supposés fournis en % (ex: 3.25), puis convertis en taux (0.0325).
    - La volatilité sigma retournée est une volatilité normale "en taux" puis convertie en bps.

    Paramètres
    ----------
    price : float
        Prix de marché (PV) de l'instrument.
    forward : float
        Taux forward (swap rate forward ou forward caplet), exprimé en %.
    strike : float
        Strike, exprimé en %.
    expiry : float
        Maturité/échéance en années (T).
    notional : float
        Nominal de l'instrument.
    annuity : float
        Facteur d'annuité (souvent somme des DF * accruals pour swaptions),
        ou DF*yearfrac pour un caplet, selon ta convention de pricing.

    Retourne
    --------
    float
        Volatilité implicite normale en basis points (bps).
    """
    # Conversion : % -> taux décimaux
    forward = forward / 100.0
    strike = strike / 100.0

    def bachelier_price(sigma):
        """
        Prix Bachelier (normal model) pour une option de type call sur taux :
          PV = annuity * notional * [ (F-K) * N(d) + sigma*sqrt(T) * n(d) ]
        avec d = (F-K)/(sigma*sqrt(T))

        Remarque : ici on ne gère pas explicitement payer/receiver (w),
        on est sur la formule "call" standard (payer swaption).
        """
        # Sécurité : vol <= 0 => prix nul (pour éviter divisions par zéro)
        if sigma <= 0:
            return 0.0

        # d de Bachelier
        d = (forward - strike) / (sigma * np.sqrt(expiry))

        # Prix modèle (PV)
        price_model = annuity * notional * (
            (forward - strike) * norm.cdf(d)
            + sigma * np.sqrt(expiry) * norm.pdf(d)
        )
        return float(price_model)

    def objective(sigma):
        """
        Équation d'inversion : on cherche sigma tel que
        bachelier_price(sigma) = price  <=> objective(sigma) = 0
        """
        return bachelier_price(sigma) - price

    # Bornes raisonnables pour sigma en "taux" (unités décimales).
    # 1e-6 évite les divisions par zéro ; 5.0 est volontairement large.
    try:
        sigma_normal = brentq(objective, 1e-6, 5.0)
    except ValueError as e:
        sigma_normal = np.nan
        print(f"Warning: Impossible de résoudre la vol implicite : {e}")

    # Conversion taux -> bps : 1.0 = 10000 bps
    return sigma_normal * 10000.0
