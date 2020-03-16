import datetime as dt

import pytest

from sova.asset import StockOption, Deposit, Stock
from sova.black_scholes import OptionType
from sova.portfolio import Portfolio, Trade

precision = 1e-5

expiry = dt.datetime(2021, 1, 1, 0, 0)
market_time = dt.datetime(2020, 1, 1, 0, 0)
strike = 1.
sigma = .1
amount = 1.

mults = [.8, .9, 1, 1.1, 1.2]
opt = StockOption(amount=amount,
                  expiry=expiry,
                  strike=strike,
                  sigma=sigma,
                  option_type=OptionType.CALL)


@pytest.mark.parametrize('mult', mults)
def test_portfolio_price(mult):
    stock_price = mult * strike
    n = 7
    trades = list(map(lambda a: Trade(market_time, stock_price, a),
                      [opt, Deposit(1), Deposit(-1), Stock(10), Stock(-10)]))
    portfolio = Portfolio(trades=n * trades)

    assert pytest.approx(n * opt.current_price(stock_price, market_time), precision) == portfolio.current_price(
        stock_price,
        market_time)


@pytest.mark.parametrize('mult', mults)
def test_portfolio_delta(mult):
    stock_price = mult * strike
    n = 7
    trades = list(map(lambda a: Trade(market_time, stock_price, a), [opt, Deposit(1), Stock(10), Stock(-10)]))
    portfolio = Portfolio(trades=n * trades)

    assert pytest.approx(n * opt.current_delta(stock_price, market_time), precision) == portfolio.current_delta(
        stock_price,
        market_time)


@pytest.mark.parametrize('mult', mults)
def test_portfolio_is_delta_neutral_after_hedge(mult):
    stock_price = mult * strike
    n = 7
    trades = list(map(lambda a: Trade(market_time, stock_price, a),
                      [opt, Deposit(1), Stock(10)]))
    portfolio = Portfolio(trades=n * trades)
    portfolio.hedge_delta(stock_price, market_time)

    assert pytest.approx(portfolio.current_delta(stock_price, market_time), precision) == 0


@pytest.mark.parametrize('mult', mults)
def test_portfolio_price_does_not_change_after_hedge(mult):
    stock_price = mult * strike
    n = 7
    trades = list(map(lambda a: Trade(market_time, stock_price, a),
                      [opt, Deposit(1), Stock(10)]))
    portfolio = Portfolio(trades=n * trades)
    portfolio_pre_hedge_price = portfolio.current_price(stock_price, market_time)
    portfolio_after_hedge_price = portfolio.current_price(stock_price, market_time)

    assert pytest.approx(portfolio_pre_hedge_price, precision) == portfolio_after_hedge_price


@pytest.mark.parametrize('mult', mults)
def test_portfolio_hedges_has_corresponding_amounts(mult):
    stock_price = mult * strike
    n = 7
    trades = list(map(lambda a: Trade(market_time, stock_price, a),
                      [opt, Deposit(1), Stock(10)]))
    portfolio = Portfolio(trades=n * trades)
    deposit_hedge, stock_hedge = portfolio.hedge_delta(stock_price, market_time)

    assert pytest.approx(-1 * deposit_hedge.amount / stock_price, precision) == stock_hedge.amount
