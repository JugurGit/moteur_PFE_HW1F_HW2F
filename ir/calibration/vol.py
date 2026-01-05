from scipy.optimize import minimize, brentq
import numpy as np
from scipy.stats import norm 
import itertools


# Bachelier (normal) vol from price
def black_normal_vol(price, forward, strike, expiry, notional, annuity):
    """
    Computes the normal (Bachelier) implied volatility (in basis points) from a swaption or caplet price.

    Parameters
    ----------
    price : float
        Market price of the swaption or caplet.
    forward : float
        Forward swap rate.
    strike : float
        Swaption strike rate.
    expiry : float
        Time to expiry in years.
    notional : float
        Notional amount of the swaption or caplet.
    annuity : float
        Annuity factor for the swaption or caplet (DF x Year Frac.).

    Returns
    -------
    float
        Implied normal volatility in basis points (bps).
    """
    forward = forward / 100  # convert percentage to rate units
    strike = strike / 100    # convert percentage to rate units

    def bachelier_price(sigma):
        if sigma <= 0:
            return 0.0
        d = (forward - strike) / (sigma * np.sqrt(expiry))
        price_model = annuity * notional * ((forward - strike) * norm.cdf(d) + sigma * np.sqrt(expiry) * norm.pdf(d))
        return price_model

    def objective(sigma):
        return bachelier_price(sigma) - price

    # Reasonable bounds for normal vols (in rate units)
    try:
        sigma_normal = brentq(objective, 1e-6, 5.0)
        
    except ValueError as e:
        sigma_normal = np.nan
        print(f"Warning: Could not solve for vol: {e}")
    return sigma_normal * 10000  # convert to bps