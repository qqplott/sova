import numpy as np

from market_data import generate_market_data
from simulation import HedgeSimulation


def test_simulation():
    a0 = 1e-5
    n_obs = 500
    p_arr = np.array([.01])
    q_arr = np.array([.002])

    md = generate_market_data(a0, p_arr, q_arr, n_obs=n_obs)
    sim = HedgeSimulation()
    sim_df = sim.run_simulation(md)

    print(sim_df.head())
