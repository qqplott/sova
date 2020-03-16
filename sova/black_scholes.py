from enum import Enum

import numpy as np
import scipy.stats as ss
import sympy as sy
from sympy.stats import Normal, cdf


class OptionType(Enum):
    CALL = 1
    PUT = -1


def put_option_price_sym(S, K, T, r, sigma):
    # S: spot price
    # K: strike price
    # T: time to maturity
    # r: interest rate
    # sigma: volatility of underlying asset

    N = Normal('x', 0.0, 1.0)

    S = float(S)

    d1 = (sy.ln(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sy.sqrt(T))
    d2 = (sy.ln(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * sy.sqrt(T))

    put_price = K * sy.exp(-r * T) * cdf(N)(-d2) - S * cdf(N)(-d1)

    return put_price


def black_scholes_price(payoff: OptionType,
                        s_0: float,
                        k: float,
                        t: float,
                        r: float,
                        sigma: float) -> float:
    d1 = (np.log(s_0 / k) + (r + sigma ** 2 / 2) * t) / (sigma * np.sqrt(t))
    d2 = (np.log(s_0 / k) + (r - sigma ** 2 / 2) * t) / (sigma * np.sqrt(t))

    # By some reason payoff == OptionType.CALL does not work
    if payoff.value == OptionType.CALL.value:
        return s_0 * ss.norm.cdf(d1) - k * np.exp(-r * t) * ss.norm.cdf(d2)
    else:
        return k * np.exp(-r * t) * ss.norm.cdf(-d2) - s_0 * ss.norm.cdf(-d1)


def black_scholes_delta(payoff: OptionType = OptionType.CALL,
                        s_0: float = 100.,
                        k: float = 100.,
                        t: float = 1.,
                        r: float = 0.1,
                        sigma: float = 0.2) -> float:
    d1 = (np.log(s_0 / k) + (r + sigma ** 2 / 2) * t) / (sigma * np.sqrt(t))

    if payoff == OptionType.CALL:
        return ss.norm.cdf(d1)
    else:
        return ss.norm.cdf(d1) - 1


def black_scholes_gamma(s_0: float = 100.,
                        k: float = 100.,
                        t: float = 1.,
                        r: float = 0.1,
                        sigma: float = 0.2) -> float:
    d1 = (np.log(s_0 / k) + (r + sigma ** 2 / 2) * t) / (sigma * np.sqrt(t))
    return ss.norm.pdf(d1) / s_0 / np.sqrt(t)
