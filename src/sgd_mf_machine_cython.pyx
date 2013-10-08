'''
Matrix factorization based on SGD
'''

cimport numpy as np
import bottleneck as bn

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
        cdef int D = params.p_D
        cdef int M = params.p_M
        cdef int N = params.p_N

        cdef np.ndarray[np.float64_t, ndim=1] itemlist

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

                err = obsv - model.predict(user, item, P, Q)
                rmse_err += err * err
                for i in xrange(D):
                    P[user, i] += sgd_gamma * (err * Q[item, i]
                                               - sgd_lambda * P[user, i])
                    Q[item, i] += sgd_gamma * (err * P[user, i]
                                               - sgd_lambda * Q[item, i])

            # Update parameters
            sgd_gamma *= params.p_step_dec

            # Possible convergence check
            print "Iteration Time: " + str(time.clock() - start)
            print math.sqrt(rmse_err / len(data.ratings))

            # Getting top-N recommendation for every user
            start = time.clock()

            itemlist = np.empty(N, dtype=np.float64)
            for i in xrange(M):
                for j in xrange(N):
                    itemlist[j] = P[i, ].dot(Q[j, ])
                bn.partsort(itemlist, 10)

            print "Top-N Time: " + str(time.clock() - start)
