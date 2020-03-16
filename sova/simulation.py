import datetime as dt
from typing import Optional, Dict

import numpy as np
import pandas as pd

from sova.asset import StockOption
from sova.market_data import generate_market_data
from sova.portfolio import Portfolio, Trade


class HedgeSimulation:
    def __init__(self, portfolio: Optional[Portfolio] = None):
        self.portfolio = portfolio

    def _make_result_dt_row(self, curr_mkt_time: dt.datetime, curr_mkt: pd.Series) -> Dict:
        hedge_deposit, hedge_asset = self.portfolio.hedge_delta(curr_mkt.asset_price,
                                                                curr_mkt_time,
                                                                curr_mkt.risk_free_rate)
        curr_pv = self.portfolio.current_price(curr_mkt.asset_price,
                                               curr_mkt_time,
                                               curr_mkt.risk_free_rate)
        curr_delta = self.portfolio.current_delta(curr_mkt.asset_price,
                                                  curr_mkt_time,
                                                  curr_mkt.risk_free_rate)
        return {
            'asset_price': curr_mkt.asset_price,
            'porfolio_pv': curr_pv,
            'portfolio_delta': curr_delta,
            'hedge_asset_amount': hedge_asset.asset.amount,
            'hedge_deposit_amount': hedge_deposit.asset.amount
        }

    def run_simulation(self, market_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        market_data = market_data if market_data is not None else generate_market_data(a0=1e-4,
                                                                                       p_arr=np.array([.01]),
                                                                                       q_arr=np.array([.02]),
                                                                                       n_obs=366)
        curr_mkt = market_data.iloc[0]
        curr_mkt_time = curr_mkt.name.to_pydatetime()

        if self.portfolio is None:
            opt = StockOption(amount=1,
                              expiry=curr_mkt_time,
                              strike=curr_mkt.asset_price * curr_mkt.risk_free_rate)
            initial_assets = [opt]
            self.portfolio = Portfolio(list(map(lambda a: Trade(curr_mkt_time, curr_mkt.asset_price, a),
                                                initial_assets)))

        hedge_result_dict = {curr_mkt_time: self._make_result_dt_row(curr_mkt_time.to_pydatetime(), curr_mkt)
                             for curr_mkt_time, curr_mkt in market_data.iterrows()}
        return pd.DataFrame.from_dict(hedge_result_dict, orient='index')
