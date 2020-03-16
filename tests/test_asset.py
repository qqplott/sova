import datetime as dt
from itertools import product

import numpy as np
import pytest

from asset import StockOption, Stock, Deposit
from black_scholes import OptionType

precision = 1e-5

expiry = dt.datetime(2021, 1, 1, 0, 0)
market_time = dt.datetime(2020, 1, 1, 0, 0)
strike = 1.
sigma = .1
amount = 1.

mults = [.8, .9, 1, 1.1, 1.2]
test_parameters = list(product([OptionType.CALL, OptionType.PUT], mults))


@pytest.mark.parametrize('option_type, mult', test_parameters)
def test_option_bs_price(option_type, mult):
    opt = StockOption(amount=amount,
                      expiry=expiry,
                      strike=strike,
                      sigma=sigma,
                      option_type=option_type)

    asset_price = mult * strike

    price = opt.current_price(asset_price, market_time)
    print(f'BS price: {price}')
    assert price > 0


@pytest.mark.parametrize('option_type, mult', test_parameters)
def test_option_with_bigger_amount_have_higher_price(option_type, mult):
    opt1 = StockOption(amount=amount,
                       expiry=expiry,
                       strike=strike,
                       sigma=sigma,
                       option_type=option_type)
    opt2 = StockOption(amount=2 * amount,
                       expiry=expiry,
                       strike=strike,
                       sigma=sigma,
                       option_type=option_type)

    asset_price = mult * strike

    price1 = opt1.current_price(asset_price, market_time)
    price2 = opt2.current_price(asset_price, market_time)

    print(f'BS price 1: {price1}')
    print(f'BS price 2: {price2}')

    assert pytest.approx(2 * price1, precision) == price2


# @pytest.mark.parametrize('option_type', [OptionType.CALL, OptionType.PUT])
def test_option_with_bigger_tenor_have_higher_price():
    opt1 = StockOption(amount=amount,
                       expiry=expiry,
                       strike=strike,
                       sigma=sigma,
                       option_type=OptionType.CALL)
    opt2 = StockOption(amount=amount,
                       expiry=expiry + dt.timedelta(days=365),
                       strike=strike,
                       sigma=sigma,
                       option_type=OptionType.CALL)

    price1 = opt1.current_price(strike, market_time)
    price2 = opt2.current_price(strike, market_time)

    print(f'BS price 1: {price1}')
    print(f'BS price 2: {price2}')

    assert price1 < price2


@pytest.mark.parametrize('mult', mults)
def test_call_put_parity(mult):
    risk_free_rate = .05

    call_opt = StockOption(amount=amount,
                           expiry=expiry,
                           strike=strike,
                           sigma=sigma,
                           option_type=OptionType.CALL)
    put_opt = StockOption(amount=amount,
                          expiry=expiry,
                          strike=strike,
                          sigma=sigma,
                          option_type=OptionType.PUT)

    t = (expiry - market_time).total_seconds() / StockOption.seconds_in_year
    asset_price = mult * strike

    call_price = call_opt.current_price(asset_price, market_time, risk_free_rate)
    put_price = put_opt.current_price(asset_price, market_time, risk_free_rate)

    assert pytest.approx(call_price - put_price, precision) == amount * (asset_price - strike * np.exp(-risk_free_rate * t))


@pytest.mark.parametrize('mult', mults)
def test_stock_price(mult):
    stock_amount = 7
    stock_price = mult * strike
    stock = Stock(stock_amount)
    price = stock.current_price(stock_price, market_time)

    assert pytest.approx(stock_amount * stock_price, precision) == price


@pytest.mark.parametrize('mult', mults)
def test_stock_delta(mult):
    stock_amount = 7
    stock_price = mult * strike
    stock = Stock(stock_amount)
    delta = stock.current_delta(stock_price, market_time)

    assert pytest.approx(stock_amount, precision) == delta


@pytest.mark.parametrize('mult', mults)
def test_deposit_price(mult):
    deposit_amount = 7
    deposit = Deposit(deposit_amount)
    asset_price = strike * mult
    price = deposit.current_price(asset_price, market_time)

    assert pytest.approx(deposit_amount, precision) == price


@pytest.mark.parametrize('mult', mults)
def test_deposit_delta(mult):
    deposit_amount = 7
    deposit = Deposit(deposit_amount)
    asset_price = strike * mult
    delta = deposit.current_delta(asset_price, market_time)

    assert pytest.approx(0, precision) == delta
