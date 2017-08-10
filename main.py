#!/usr/bin/env python

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gamma


def realized_quantity(fun):
    """Applies the function 'fun' to each day separately"""
    return intraday_returns.groupby(pd.TimeGrouper("1d")).apply(fun)[index]

# TODO add subsampling
# TODO add more realized quantities (kernel, intraday range, etc.)

if __name__ == "__main__":

    # Settings
    asset = "IBM"
    sampling = "cts"
    trading_seconds = 23400
    avg_sampling_frequency = 300
    original_sampling_frequency = 60  # From the process_data.py file
    M = trading_seconds / original_sampling_frequency

    # Load data and store the intraday returns
    data = pd.read_hdf("in/" + asset + "_" + sampling + ".h5", "table")
    intraday_returns = data.groupby(pd.TimeGrouper("1d")).apply(lambda x: np.log(x / x.shift(1))).dropna()

    # Index of all days
    index = data.groupby(pd.TimeGrouper("1d")).first().dropna().index

    # Some constants
    mu_1 = np.sqrt((2 / np.pi))
    mu_43 = 2 ** (2 / 3) * gamma(7 / 6) * gamma(1 / 2) ** (-1)

    # First and last price of each day
    prices_open = data.resample('D').first()[index]
    prices_close = data.resample('D').last()[index]

    # Return (close-to-close and open-to-close
    r_cc = pd.Series(np.log(prices_close / prices_close.shift(1)), name='r_cc')
    r_oc = pd.Series(np.log(prices_close / prices_open), name='r_oc')

    # Realized Variance (Andersen and Bollerslev, 1998)
    rv = realized_quantity(lambda x: (x ** 2).sum())

    # Realized absolute variation (Forsberg and Ghysels, 2007)
    rav = mu_1 ** (-1) * M ** (-.5) * realized_quantity(lambda x: x.abs().sum())

    # Realized bipower variation (Barndorff-Nielsen and Shephard; 2004, 2006)
    bv = mu_1 ** (-2) * realized_quantity(lambda x: (x.abs() * x.shift(1).abs()).sum())

    # Standardized tri-power quarticity (see e.g. Forsberg & Ghysels, 2007)
    tq = M * mu_43 ** (-3) * realized_quantity(
        lambda x: (x.abs() ** (4 / 3) * x.shift(1).abs() ** (4 / 3) * x.shift(2).abs() ** (4 / 3)).sum())

    # Jump test by Huang and Tauchen (2005)
    j = (np.log(rv) - np.log(bv)) / \
        ((mu_1 ** -4 + 2 * mu_1 ** -2 - 5) / (M * tq * bv ** -2)) ** 0.5
    jump = j.abs() >= stats.norm.ppf(0.999)

    # Separate continuous and discontinuous parts of the quadratic variation
    iv = pd.Series(0, index=index)
    iv[jump] = bv[jump] ** 0.5
    iv[~jump] = rv[~jump] ** 0.5

    jv = pd.Series(0, index=index)
    jv[jump] = rv[jump] ** 0.5 - bv[jump] ** 0.5
    jv[jv < 0] = 0

    # Realized Semivariance (Barndorff-Nielsen, Kinnebrock and Shephard, 2010)
    rv_m = realized_quantity(lambda x: (x ** 2 * (x < 0)).sum())
    rv_p = realized_quantity(lambda x: (x ** 2 * (x > 0)).sum())

    # Signed jump variation (Patton and Sheppard, 2015)
    sjv = rv_p ** 0.5 - rv_m ** 0.5
    sjv_p = sjv * (sjv > 0)
    sjv_m = sjv * (sjv < 0)

    # Realized Skewness and Kurtosis  (see, e.g. Amaya, Christoffersen, Jacobs and Vasquez, 2015)
    rm3 = realized_quantity(lambda x: (x ** 3).sum())
    rm4 = realized_quantity(lambda x: (x ** 4).sum())
    rs = np.sqrt(M) * rm3 / rv ** (3 / 2)
    rk = M * rm4 / rv ** 2

    # Export data
    out = pd.concat([r_cc, r_oc, rav, rv ** .5, bv ** .5, rv_m ** 0.5, rv_p ** 0.5,
                     iv, jv, sjv, sjv_p, sjv_m, rs, rk], axis=1)
    out.columns = np.array(['r_cc', 'r_oc', 'rav', 'rvol', 'bvol', 'rvol_m', 'rvol_p',
                            'ivol', 'jvol', 'sjv', 'sjv_p', 'sjv_m', 'rs', 'rk'])
    out.to_csv('out/realized_quantities_' + asset + "_" + sampling + ".csv")
