'''
This class defines the latent model itself, but does nothing about
training and testing, those should be defined in a separate class.
'''

import numpy as np
cimport numpy as np

class LatentModel:
    def __init__(self, params):
        self.minval = 1e-100
        self.maxval = 1e100
        self.M = params.p_M
        self.N = params.p_N
        self.D = params.p_D

    def reset(self, np.ndarray[np.float64_t, ndim=2] P,
              np.ndarray[np.float64_t, ndim=2] Q):
        # Randomly initiate matrixes with values from 0 to 1
        P = np.random.random_sample((self.M, self.D))
        Q = np.random.random_sample((self.N, self.D))

    '''Predict score for user i(0:) and item j(0:)'''
    def predict(self, int i, int j,
                np.ndarray[np.float64_t, ndim=2] P,
                np.ndarray[np.float64_t, ndim=2] Q):
        cdef double pred_rating = P[i, ].dot(Q[j, ])
        pred_rating = max(pred_rating, self.minval)
        pred_rating = min(pred_rating, self.maxval)
        return pred_rating
