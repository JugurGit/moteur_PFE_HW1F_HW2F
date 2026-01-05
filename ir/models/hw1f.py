import pandas as pd
import numpy as np


class HullWhiteModel:
    """
    Hull-White one-factor short rate model (no simulation).
    
    This class implements the analytical formulas for the 
    extended Vasicek/Hull-White model:
    
        dr(t) = a * (θ(t) - r(t)) dt + σ dW(t)
    
    where:
        a     = mean reversion speed
        σ     = volatility of the short rate
        θ(t)  = time-dependent drift fitted to the initial curve
    
    Attributes
    ----------
    curve : Curve
        Instance representing the initial discount curve P(0, T).
    parameters : dict
        Dictionary containing model parameters:
            - 'a' : float, mean reversion speed
            - 'sigma' : float, volatility
            - 'r0' : float, initial short rate
    a : float
        Mean reversion speed.
    sigma : float
        Volatility parameter.
    r0 : float
        Initial short rate.
    """

    def __init__(self, curve, parameters=None):
        """
        Initialize the Hull-White model with a given discount curve and parameters.

        Parameters
        ----------
        curve : Curve
            Discount curve used to fit θ(t) and compute forwards.
        parameters : dict, optional
            Dictionary with 'a', 'sigma', and 'r0'. If None, defaults are used: {'a': 0.01, 'sigma': 0.01, 'r0': curve.inst_forward_rate(0)}
        """

        self.curve = curve

        # Default parameters
        defaults = {'a': 0.01, 'sigma': 0.01, 'r0': curve.inst_forward_rate(0)}
        if parameters is None: parameters = {}
        self.parameters = {'a': parameters.get('a', defaults['a']),'sigma': parameters.get('sigma', defaults['sigma']),'r0': parameters.get('r0', defaults['r0'])}

    def inst_forward_rate(self, t):
        """
        Compute the instantaneous forward rate f(0, t).

        Parameters
        ----------
        t : float
            Time in years.

        Returns
        -------
        float
            Instantaneous forward rate at time t.
        """
        return self.curve.inst_forward_rate(t)

    def discount_factor(self, t):
        """
        Compute the discount factor P(0, t).

        Parameters
        ----------
        t : float
            Time in years.

        Returns
        -------
        float
            Discount factor for maturity t.
        """
        return self.curve.discount(t)
    
    def forward_rate(self, T1, T2):
        """
        Compute the simple forward rate F(0; T1, T2) implied by the discount curve. 
        Parameters
        ----------      
        T1 : float
            Start time of the forward rate.
        T2 : float
            End time of the forward rate.
        Returns 
        -------
        float
            Forward rate between T1 and T2.
        """
        return self.curve.forward_rate(T1, T2)

    def alpha(self, t):
        """
        Compute α(t), the deterministic shift function in Hull–White.

        Formula:
            α(t) = f(0, t) + (σ² / (2a²)) * (1 - e^{-a t})²

        Parameters
        ----------
        t : float
            Time in years.

        Returns
        -------
        float
            α(t) value.
        """
        a = self.parameters['a']
        sigma = self.parameters['sigma']
        fwd = self.inst_forward_rate(t)
        return fwd + (sigma**2) / (2 * a**2) * (1 - np.exp(-a * t))**2

    def B(self, t, T):
        """
        Compute B(t, T) function used in bond pricing.

        Formula:
            B(t, T) = (1 - e^{-a (T - t)}) / a

        Parameters
        ----------
        t : float
            Start time in years.
        T : float
            Maturity time in years.

        Returns
        -------
        float
            B(t, T) value.
        """
        a = self.parameters['a']
        return (1 - np.exp(-a * (T - t))) / a

    def A(self, t, T):
        """
        Compute A(t, T) function used in zero-coupon bond pricing.

        Formula:
            A(t, T) = [P(0, T) / P(0, t)] * exp(B(t, T) * f(0, t) - 
                       (σ² / (4a)) * (1 - e^{-2a t}) * B(t, T)²)

        Parameters
        ----------
        t : float
            Start time in years.
        T : float
            Maturity time in years.

        Returns
        -------
        float
            A(t, T) value.
        """
        a = self.parameters['a']
        sigma = self.parameters['sigma']
        P_t = self.discount_factor(t)
        P_T = self.discount_factor(T)
        fwd = self.inst_forward_rate(t)
        B = self.B(t, T)
        return (P_T / P_t) * np.exp(
            B * fwd - (sigma**2 / (4 * a)) * (1 - np.exp(-2 * a * t)) * B**2
        )

    def short_rate(self, t, z=None):
        """
        Compute the short rate r(t) under the risk-neutral measure 
        using the exact distribution.

        Distribution:
            r(t) ~ Normal(mean = E[r(t)], variance = V[r(t)])

        Parameters
        ----------
        t : float
            Time in years.
        z : float, optional
            Standard normal draw. If None, one is generated.

        Returns
        -------
        float
            Simulated short rate at time t.
        """
        if z is None:
            z = np.random.normal()

        r0 = self.parameters['r0']
        a = self.parameters['a']
        sigma = self.parameters['sigma']
        V = (sigma**2 / (2 * a)) * (1 - np.exp(-2 * a * t))
        E = r0 * np.exp(-a * t) + self.alpha(t) - np.exp(-a * t) * self.alpha(0)
        return E + np.sqrt(V) * z


class HullWhiteSimulation:
    """
    Monte Carlo simulation engine for the Hull–White one-factor model.

    Provides:
        - Exact simulation of r(T) at a single maturity (no path generation)
        - Euler–Maruyama path simulation under the risk-neutral measure
        - Analytical validation of simulated mean and variance

    Attributes
    ----------
    model : HullWhiteModel
        Hull–White model instance providing parameters and curve.
    n_paths : int
        Number of Monte Carlo paths.
    n_steps : int
        Number of time steps for Euler path simulation.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, model: HullWhiteModel, n_paths=10**5, n_steps=100, seed=2025):
        """
        Initialize the Hull–White simulation engine.

        Parameters
        ----------
        model : HullWhiteModel
            Hull–White model instance.
        n_paths : int, optional
            Number of Monte Carlo paths (default: 100,000).
        n_steps : int, optional
            Number of steps for Euler path simulation (default: 100).
        seed : int, optional
            Random seed for reproducibility (default: 2025).
        """
        self.model = model
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.seed = np.random.seed(seed)

    def simulate_short_rate_direct(self, T):
        """
        Simulate r(T) using the exact analytical distribution 
        under the risk-neutral measure.

        Parameters
        ----------
        T : float
            Simulation horizon in years.

        Returns
        -------
        ndarray
            Array of simulated short rates (n_paths,).
        """
        z = np.random.normal(size=self.n_paths)
        r = np.array([self.model.short_rate(T, z=z_i) for z_i in z])
        return r


class HullWhiteCurveBuilder:
    """
    Hull–White curve builder that provides both analytical formulas and Monte Carlo 
    simulation utilities for pricing zero-coupon bonds, discount factors, 
    forward rates, and long-term rates, using a pre-built discount curve.

    Attributes
    ----------
    model : HullWhiteModel
        Hull–White model instance constructed from the provided curve and parameters.
    sim : HullWhiteSimulation
        Monte Carlo simulation engine built from the Hull–White model.
    curve : Curve
        Pre-initialized discount curve used for forwards and short rate calculations.
    """

    def __init__(self, curve, params=None, n_paths=10**5, n_steps=100, seed=2025, smooth=1e-7):
        """
        Initialize the Hull–White curve builder using a pre-built Curve instance and 
        Hull–White model parameters.

        Parameters
        ----------
        Curve : Curve
            Pre-initialized discount curve instance containing times to maturity 
            and discount factors.
        params : dict, optional
            Dictionary containing Hull–White model parameters (optional):
                - 'a' : float, mean reversion speed
                - 'sigma' : float, volatility
                - 'r0' : float, initial short rate
            If None, defaults are used: {'a': 0.01, 'sigma': 0.01, 'r0': curve.inst_forward_rate(0)}
        n_paths : int, optional
            Number of Monte Carlo paths (default: 100,000).
        n_steps : int, optional
            Number of discretization steps per path (default: 100).
        seed : int, optional
            Random seed for reproducibility (default: 2025).
        smooth : float, optional
            Smoothing parameter for the discount curve (not used if Curve is already initialized).

        Workflow
        --------
        1. Use the provided Curve instance for discount factors and instantaneous forwards.
        2. Build the Hull–White model using the curve and provided parameters.
        3. Initialize the Monte Carlo simulation engine for short rate paths.
        """
        self.curve = curve
        self.model = HullWhiteModel(self.curve, params)
        self.sim = HullWhiteSimulation(self.model, n_paths=n_paths, n_steps=n_steps, seed=seed)


    def short_rate(self, t):
        """
        Simulate the short rate r(t) at a single time t using the exact distribution.

        Parameters
        ----------
        t : float
            Time in years.

        Returns
        -------
        ndarray
            Array of simulated short rates (n_paths,).
        """
        return self.sim.simulate_short_rate_direct(t)


    def zero_coupon_bond(self, t, T):
        """
        Price a zero-coupon bond analytically under the risk-neutral 

        Formula:
            P(t, T) = A(t, T) * exp(-B(t, T) * r(t))

        Parameters
        ----------
        t : float
            Current time in years.
        T : float
            Bond maturity in years.

        Returns
        -------
        ndarray
            Bond price distribution.
        """
        r_t = self.sim.simulate_short_rate_direct(t)

        A = self.model.A(t, T)
        B = self.model.B(t, T)
        price = A * np.exp(-B * r_t)
        return price

