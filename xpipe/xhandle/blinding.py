"""
Blinding Schemes
"""

import numpy as np


class BlindLWB(object):
    def __init__(self, seed=5):
        """
        DeltaSigma Blinding for Lensing Without Borders


        Draw alpha and beta randomly from [0.8; 1.2] and then multiply the DeltaSigma by f(r):

            f(r) = 1/9 * ((beta - alpha) * r + 10 * alpha - beta)

        """
        self.rng = np.random.RandomState(seed=seed)
        self.alpha = None
        self.beta = None

    def f(self, r):
        """ calculate f(r)"""
        val = 1. / 9. * ((self.beta - self.alpha) * r + 10. * self.alpha - self.beta)
        return val

    def draw(self):
        self.alpha = self.rng.uniform(low=0.8, high=1.2)
        self.beta = self.rng.uniform(low=0.8, high=1.2)
