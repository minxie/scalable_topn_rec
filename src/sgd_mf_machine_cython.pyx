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
    def __init__(self, model, data):
        MFMachine.__init__(self, model, data)

    def train(self, params):
        np.seterr(all='raise')

        cdef double sgd_gamma = params.p_gamma
        cdef double sgd_lambda = params.p_lambda

        cdef double rmse_err = -1.0
        cdef int user = -1
        cdef int item = -1
        cdef double obsv = -1.0
        cdef double err = -1.0
        cdef int D = 0

        processing_order = range(len(self._data.ratings))
        for tr_iter in xrange(params.p_max_i):
            start = time.clock()
            
            # Shuffling order of processing the tuples
            random.shuffle(processing_order)

            # Run through all training examples
            rmse_err = 0.0
            for idx in processing_order:
                user = self._data.ratings[idx][0]
                item = self._data.ratings[idx][1]
                obsv = self._data.ratings[idx][2]
                err = 0.0

                err = obsv - self._model.predict(user, item)
                rmse_err += err * err
                D = params.p_D
                for i in xrange(D):
                    self._model.P[user, i] += sgd_gamma * (err * self._model.Q[item, i]
                                                           - sgd_lambda * self._model.P[user, i])
                    self._model.Q[item, i] += sgd_gamma * (err * self._model.P[user, i]
                                                           - sgd_lambda * self._model.Q[item, i])

            # Update parameters
            sgd_gamma *= params.p_step_dec

            # Possible convergence check
            print "Iteration Time: " + str(time.clock() - start)
            print math.sqrt(rmse_err / len(self._data.ratings))
