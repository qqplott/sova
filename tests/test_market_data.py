import numpy as np

from sova.market_data import generate_market_data


def test_market_data_generator():
    md = generate_market_data(.01, np.array([.1]), np.array([.2]))

    print(md.head())
