import datetime as dt
from dataclasses import dataclass
from typing import List, Tuple, Callable

from sova.asset import AbstractAsset, Deposit, Stock


@dataclass
class Trade:
    trade_time: dt.datetime
    stock_price: float
    asset: AbstractAsset


class Portfolio:
    def __init__(self, trades: List[Trade]):
        self.trades = trades

    def trade(self, trade_time: dt.datetime, stock_price: float, asset: AbstractAsset):
        self.trades.append(Trade(trade_time, stock_price, asset))

    def _apply_function(self, market_time: dt.datetime, f: Callable[[Trade], float]) -> float:
        return sum(map(f, filter(lambda t: t.trade_time <= market_time, self.trades)))

    def current_price(self,
                      stock_price: float,
                      market_time: dt.datetime,
                      risk_free_rate: float = .01) -> float:
        return self._apply_function(market_time, lambda t: t.asset.current_price(stock_price=stock_price,
                                                                                 market_time=market_time,
                                                                                 risk_free_rate=risk_free_rate))

    def current_delta(self,
                      stock_price: float,
                      market_time: dt.datetime,
                      risk_free_rate: float = .01) -> float:
        return self._apply_function(market_time, lambda t: t.asset.current_delta(stock_price=stock_price,
                                                                                 market_time=market_time,
                                                                                 risk_free_rate=risk_free_rate))

    def hedge_delta(self,
                    stock_price: float,
                    market_time: dt.datetime,
                    risk_free_rate: float = .01) -> Tuple[Trade, Trade]:
        delta = self.current_delta(stock_price, market_time, risk_free_rate)
        hedge_deposit = Trade(trade_time=market_time,
                              stock_price=stock_price,
                              asset=Deposit(delta * stock_price))
        hedge_asset = Trade(trade_time=market_time,
                            stock_price=stock_price,
                            asset=Stock(-1 * delta))
        self.trades.append(hedge_deposit)
        self.trades.append(hedge_asset)
        return hedge_deposit, hedge_asset
