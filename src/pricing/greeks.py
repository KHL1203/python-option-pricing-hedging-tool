import numpy as np
from scipy.stats import norm

def calculate_d1_d2(S, K, T, r, sigma, q):
    """Calculates d1 and d2 for Black-Scholes formulas."""
    d1 = ((np.log(S/K) + (r-q + 0.5*sigma**2) * T))/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return d1, d2

def delta(S, K, T, r, sigma, q, option_type):
    """Calculates the Delta of an option."""
    d1, _ = calculate_d1_d2(S, K, T, r, sigma, q)
    if option_type.lower() == "call":
        return np.exp(-q*T) * norm.cdf(d1)
    elif option_type.lower() == "put":
        return np.exp(-q*T) * (norm.cdf(d1) - 1.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

def gamma(S, K, T, r, sigma, q):
    """Calculates the Gamma of an option."""
    d1, _ = calculate_d1_d2(S, K, T, r, sigma, q)
    return (np.exp(-q*T) * norm.pdf(d1)) / (S * sigma * np.sqrt(T))

def vega(S, K, T, r, sigma, q):
    """Calculates the Vega of an option."""
    d1, _ = calculate_d1_d2(S, K, T, r, sigma, q)
    return (S * np.exp(-q*T) * norm.pdf(d1) * np.sqrt(T)) * 0.01  # Multiplied by 0.01 to represent change per 1% vol

def theta(S, K, T, r, sigma, q, option_type):
    """Calculates the Theta of an option."""
    d1, d2 = calculate_d1_d2(S, K, T, r, sigma, q)
    term1 = -(S * sigma * norm.pdf(d1)) / (2 * np.sqrt(T))

    if option_type.lower() == "call":
        term2 = r * K * np.exp(-r * T) * norm.cdf(d2)
        return term1 - term2
    elif option_type.lower() == "put":
        term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
        return term1 + term2
    else:
        raise ValueError("option_type must be 'call' or 'put'")

def rho(S, K, T, r, sigma, q, option_type):
    """Calculates the Rho of an option."""
    _, d2 = calculate_d1_d2(S, K, T, r, sigma, q)
    if option_type.lower() == "call":
        return (K * T * np.exp(-r * T) * norm.cdf(d2)) * 0.01
    elif option_type.lower() == "put":
        return (-K * T * np.exp(-r * T) * norm.cdf(-d2)) * 0.01
    else:
        raise ValueError("option_type must be 'call' or 'put'")