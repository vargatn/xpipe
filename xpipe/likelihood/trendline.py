"""
Fit mass trend likelihood
"""

import numpy as np
from multiprocessing import Pool
import emcee
import cluster_toolkit.averaging as averaging

BADVAL = -99999

# TODO quintile trendline fit



class log_lzscaling_prob(object):
    def __init__(self, data, l_pivot=40, z_pivot=0.35):
        """logmass, c linear edges, tau, f gaussian prior (and zero cut)"""
        self.data = data
        self.l_pivot = l_pivot
        self.z_pivot = z_pivot
        self.BADVAL = BADVAL

    def __call__(self, theta):
        m0, fl, gz = theta
        lp = 0

        y = self.data["logms"]
        cov = self.data["cov"]

        larr = self.data["larr"]
        zarr = self.data["zarr"]

        model = self.calc_model(theta, larr, zarr)
        dvec = y - model
        lp += -0.5 * np.dot(np.dot(dvec.T, np.linalg.inv(cov)), dvec)

        if not np.isfinite(lp):
            lp = self.BADVAL
        return lp

    def calc_model(self, theta, larr, zarr):
        m0, fl, gz = theta
        model = m0 + np.log10((larr / self.l_pivot))*fl + np.log10((1 + zarr) / (1 + self.z_pivot))*gz

        return model


def get_confidence(flat_samples, calculator, seed=5, nsample=1000, **kwargs):

    rng = np.random.RandomState(seed)
    thetas = flat_samples[rng.randint(0, len(flat_samples), size=nsample)]

    curves = []
    for theta in thetas:
        model = calculator.calc_model(theta, **kwargs)
        curves.append(model)
    curves = np.array(curves)
    curves_16 = np.percentile(curves, 16, axis=0)
    curves_84 = np.percentile(curves, 84, axis=0)
    #     print(curves.shape)
    return curves_16, curves_84