'''
This class defines the latent model itself, but does nothing about
training and testing, those should be defined in a separate class.
'''

import numpy as np


class LatentModel:
    def __init__(self, params):
        self._params = params
        # Randomly initiate matrixes with values from -1e-3 to 1e-3
        self.P = 2e-3 * np.random.random_sample((params.p_M, params.p_D)) - 1e-3
        self.Q = 2e-3 * np.random.random_sample((params.p_N, params.p_D)) - 1e-3

    def reset(self):
        # Randomly initiate matrixes with values from -1e-3 to 1e-3
        self.P = 2e-3 * np.random.random_sample((self._params.p_M, self._params.p_D)) - 1e-3
        self.Q = 2e-3 * np.random.random_sample((self._params.p_N, self._params.p_D)) - 1e-3

    '''Predict score for user i(0:) and item j(0:)'''
    def predict(self, i, j):
        return self.P[i,].dot(self.Q[j,])
