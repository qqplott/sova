from enum import Enum

import numpy as np
import scipy.stats as ss


class OptionType(Enum):
    CALL = 1
    PUT = -1


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
