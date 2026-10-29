import numpy as np
from scipy.stats import norm

class BlackScholes:
    """
    Analytical Black-Scholes-Merton model for European options and Greeks.
    """
    def __init__(self, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0):
        """
        :param S: Spot price of the underlying asset
        :param K: Strike price
        :param T: Time to expiration in years
        :param r: Risk-free interest rate (decimal, e.g. 0.05 for 5%)
        :param sigma: Volatility of the underlying asset (decimal, e.g. 0.20 for 20%)
        :param q: Continuous dividend yield (default: 0.0)
        """
        self.S = float(S) #(Spot price): Current market price of the stock (e.g. $100).
        self.K = float(K) #(Strike price): The agreed exercise price of the contract (e.g. $105).
        self.T = max(float(T), 1e-6)  # Avoid division by zero at expiry
        self.r = float(r) #(Risk-free rate): The theoretical return of an investment with zero risk (e.g. 0.05 for 5%).
        self.sigma = max(float(sigma), 1e-6) #(Volatility): The standard deviation of the stock's returns (e.g. 0.20 for 20%).
        self.q = float(q) #(Dividend yield): The continuous dividend yield of the stock (e.g. 0.02 for 2%).
        
        self._calculate_d1_d2()

    def _calculate_d1_d2(self):
        self.d1 = (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        self.d2 = self.d1 - self.sigma * np.sqrt(self.T)

    def price(self, option_type: str = "call") -> float:
        """Calculate analytical option price."""
        if option_type.lower() == "call":
            return self.S * np.exp(-self.q * self.T) * norm.cdf(self.d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)
        elif option_type.lower() == "put":
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2) - self.S * np.exp(-self.q * self.T) * norm.cdf(-self.d1)
        else:
            raise ValueError("option_type must be either 'call' or 'put'")

    def delta(self, option_type: str = "call") -> float:
        """Delta: Sensitivity of price with respect to underlying spot price."""
        if option_type.lower() == "call":
            return np.exp(-self.q * self.T) * norm.cdf(self.d1)
        elif option_type.lower() == "put":
            return np.exp(-self.q * self.T) * (norm.cdf(self.d1) - 1.0)
        raise ValueError("option_type must be 'call' or 'put'")

    def gamma(self) -> float:
        """Gamma: Second derivative of price with respect to spot price (same for call/put)."""
        return (np.exp(-self.q * self.T) * norm.pdf(self.d1)) / (self.S * self.sigma * np.sqrt(self.T))

    def vega(self) -> float:
        """Vega: Sensitivity of price with respect to volatility (per 1% vol change)."""
        # Multiplied by 0.01 to represent change per 1% vol
        return (self.S * np.exp(-self.q * self.T) * norm.pdf(self.d1) * np.sqrt(self.T)) * 0.01

    def theta(self, option_type: str = "call") -> float:
        """Theta: Sensitivity of price with respect to time decay (per 1 day)."""
        term1 = -(self.S * self.sigma * np.exp(-self.q * self.T) * norm.pdf(self.d1)) / (2 * np.sqrt(self.T))
        
        if option_type.lower() == "call":
            term2 = - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)
            term3 = self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(self.d1)
            theta_annual = term1 + term2 + term3
        elif option_type.lower() == "put":
            term2 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2)
            term3 = - self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(-self.d1)
            theta_annual = term1 + term2 + term3
        else:
            raise ValueError("option_type must be 'call' or 'put'")
            
        return theta_annual / 365.0  # Daily Theta

    def rho(self, option_type: str = "call") -> float:
        """Rho: Sensitivity of price with respect to interest rate (per 1% rate change)."""
        if option_type.lower() == "call":
            return (self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2)) * 0.01
        elif option_type.lower() == "put":
            return (-self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2)) * 0.01
        raise ValueError("option_type must be 'call' or 'put'")