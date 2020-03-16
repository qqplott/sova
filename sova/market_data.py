import datetime as dt

import numpy as np
import pandas as pd


def generate_market_data(a0: float,
                         p_arr: np.ndarray,
                         q_arr: np.ndarray,
                         n_obs: int = 1000,
                         offset: int = 10,
                         risk_free_rate: float = 0.01) -> pd.DataFrame:
    market_start_date = dt.datetime(2020, 1, 1)

    p = len(p_arr)
    q = len(q_arr)
    p_arr = np.array([p_arr])
    q_arr = np.array([q_arr])

    eps_arr = np.array([1])
    sigma_arr = np.array([1])
    asset_arr = np.array([1])

    for i in range(1, n_obs):
        sigma = a0 + np.sum(p_arr * eps_arr[-p:] ** 2 + q_arr * sigma_arr[-q:])
        eps = np.random.normal(0, 1, 1)[0] * np.sqrt(sigma)
        asset = np.exp(eps_arr[-1]) * asset_arr[-1]

        sigma_arr = np.append(sigma_arr, sigma)
        eps_arr = np.append(eps_arr, eps)
        asset_arr = np.append(asset_arr, asset)

    return pd.DataFrame(data={
        'asset_volatility': np.sqrt(sigma_arr[offset:]),
        'asset_return': eps_arr[offset:],
        'asset_price': asset_arr[offset:],
        'market_time': [market_start_date + dt.timedelta(days=d) for d in range(offset, n_obs)],
        'risk_free_rate': risk_free_rate
    }).set_index('market_time')
