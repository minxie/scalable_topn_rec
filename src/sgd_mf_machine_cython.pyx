'''
Matrix factorization based on SGD
'''

import numpy as np
cimport numpy as np
# import bottleneck as bn
from libcpp.vector cimport vector

ctypedef vector[double].iterator diter
cdef extern from "<algorithm>" namespace "std":
    void partial_sort(diter, diter, diter)

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
        cdef double last_rmse_err = 1e6
        cdef int user = -1
        cdef int item = -1
        cdef double obsv = -1.0
        cdef double err = -1.0
        cdef int D = params.p_D
        cdef int M = params.p_M
        cdef int N = params.p_N

        cdef np.ndarray[np.float64_t, ndim=1] X

        # cdef np.ndarray[np.float64_t, ndim=1] itemlist
        cdef vector[double] itemlist
        itemlist.resize(N)

        oid = open(params.p_res_log_f_loc, 'a')
        oid.write(str(params.p_D))
        
        processing_order = range(len(data.ratings))
        start = time.clock()
        for tr_iter in xrange(params.p_max_i):
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

            print str(tr_iter) + " " + str(math.sqrt(rmse_err / len(data.ratings)))
            
            # convergence check
            if math.fabs(rmse_err - last_rmse_err) <= 1e-3:
                break
            last_rmse_err = rmse_err

        end = time.clock()
        print "Training Time: " + str(end - start)
        oid.write(' ' + str(end - start))

        # Getting top-N recommendation for every user
        start = time.clock()

        # itemlist = np.empty(N, dtype=np.float64)
        # X = P.dot(Q.T)
        for i in xrange(M):
            X = P[i, ].dot(Q.T)
            for j in xrange(N):
                # itemlist[j] = P[i, ].dot(Q[j, ])
                itemlist[j] = X[j]
            partial_sort(itemlist.begin(), itemlist.begin()+10, itemlist.end())
                # bn.partsort(X[i, ], 10)

        end = time.clock()
        print "Top-N Time: " + str(end - start)
        oid.write(' ' + str(end - start) + '\n')

        oid.close()
        
