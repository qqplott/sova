import datetime as dt
from abc import ABC, abstractmethod

from sova.black_scholes import black_scholes_delta, black_scholes_price, OptionType


class AbstractAsset(ABC):
    def __init__(self, amount):
        self.amount = amount

    @abstractmethod
    def current_price(self, stock_price: float,
                      market_time: dt.datetime,
                      risk_free_rate: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def current_delta(self, stock_price: float,
                      market_time: dt.datetime,
                      risk_free_rate: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def __add__(self, other):
        raise NotImplementedError

    def __str__(self):
        return f'{str(type(self).__name__)}(amount={self.amount})'

    def __repr__(self):
        return self.__str__()


class Deposit(AbstractAsset):
    def current_price(self, stock_price: float,
                      market_time: dt.datetime,
                      risk_free_rate=.01) -> float:
        return self.amount

    def current_delta(self, stock_price: float,
                      market_time: dt.datetime,
                      risk_free_rate=.01) -> float:
        return 0

    def __add__(self, other):
        if isinstance(other, Deposit):
            return Deposit(self.amount + other.amount)
        else:
            raise NotImplementedError('Deposit could be added only to deposit')


class Stock(AbstractAsset):
    def current_price(self,
                      stock_price: float,
                      market_time: dt.datetime,
                      risk_free_rate=.01) -> float:
        return self.amount * stock_price

    def current_delta(self, stock_price: float,
                      market_time: dt.datetime,
                      risk_free_rate=.01) -> float:
        return self.amount

    def __add__(self, other):
        if isinstance(other, Stock):
            return Stock(self.amount + other.amount)
        else:
            raise NotImplementedError('Stock could be added only to stock')


class StockOption(AbstractAsset):
    seconds_in_year = 60 * 60 * 24 * 365

    def __init__(self,
                 amount: float,
                 expiry: dt.datetime,
                 strike: float = 100.,
                 sigma: float = 0.2,
                 option_type: OptionType = OptionType.CALL):
        super().__init__(amount)
        self.expiry = expiry
        self.strike = strike
        self.sigma = sigma
        self.option_type = option_type

    def _convert_time_to_bs(self, market_time: dt.datetime):
        return (self.expiry - market_time).total_seconds() / self.seconds_in_year

    def current_price(self,
                      stock_price: float,
                      market_time: dt.datetime,
                      risk_free_rate: float = 0.01) -> float:
        if self.expiry < market_time:
            return 0
        else:
            return self.amount * black_scholes_price(payoff=self.option_type,
                                                     s_0=stock_price,
                                                     k=self.strike,
                                                     t=self._convert_time_to_bs(market_time),
                                                     r=risk_free_rate,
                                                     sigma=self.sigma)

    def current_delta(self,
                      stock_price: float,
                      market_time: dt.datetime,
                      risk_free_rate: float = 0.01) -> float:
        if self.expiry < market_time:
            return 0
        elif self.expiry < market_time:
            if self.option_type == OptionType.CALL:
                return self.amount if self.strike < stock_price else 0
            else:
                return self.amount if self.strike > stock_price else 0
        else:
            return self.amount * black_scholes_delta(payoff=self.option_type,
                                                     s_0=stock_price,
                                                     k=self.strike,
                                                     t=self._convert_time_to_bs(market_time),
                                                     r=risk_free_rate,
                                                     sigma=self.sigma)

    def __add__(self, other):
        raise NotImplementedError('Add operation is not supported for options')
