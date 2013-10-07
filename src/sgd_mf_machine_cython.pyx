'''
Matrix factorization based on SGD
'''

import numpy as np
cimport numpy as np

from mf_machine import MFMachine
import random
import sys
import math
import time


class SGDMachine(MFMachine):
    def __init__(self):
        MFMachine.__init__(self)

    def train(self, params, model, data,
              np.ndarray[np.float64_t, ndim=2] P,
              np.ndarray[np.float64_t, ndim=2] Q):
        np.seterr(all='raise')

        cdef double sgd_gamma = params.p_gamma
        cdef double sgd_lambda = params.p_lambda

        cdef double rmse_err = -1.0
        cdef int user = -1
        cdef int item = -1
        cdef double obsv = -1.0
        cdef double err = -1.0
        cdef int D = 0

        processing_order = range(len(data.ratings))
        for tr_iter in xrange(params.p_max_i):
            start = time.clock()
            
            # Shuffling order of processing the tuples
            random.shuffle(processing_order)

            # Run through all training examples
            rmse_err = 0.0
            for idx in processing_order:
                user = data.ratings[idx][0]
                item = data.ratings[idx][1]
                obsv = data.ratings[idx][2]
                err = 0.0

                err = obsv - model.predict(user, item)
                rmse_err += err * err
                D = params.p_D
                for i in xrange(D):
                    P[user, i] += sgd_gamma * (err * Q[item, i]
                                               - sgd_lambda * P[user, i])
                    Q[item, i] += sgd_gamma * (err * P[user, i]
                                               - sgd_lambda * Q[item, i])

            # Update parameters
            sgd_gamma *= params.p_step_dec

            # Possible convergence check
            print "Iteration Time: " + str(time.clock() - start)
            print math.sqrt(rmse_err / len(self._data.ratings))
