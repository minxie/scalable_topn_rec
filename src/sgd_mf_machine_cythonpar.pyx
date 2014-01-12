'''
Matrix factorization based on SGD
'''

import numpy as np
import struct
cimport numpy as np
# import bottleneck as bn
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from cpython cimport bool

ctypedef pair[double, long] par
cdef extern from "compare.h":
    cdef cppclass comp_last:
        comp_last()
        bool operator()(par, par)

ctypedef vector[double].iterator diter    
cdef extern from "<algorithm>" namespace "std":
    void partial_sort(diter, diter, diter, comp_last)

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

        cdef double a = params.p_a
        cdef double b = params.p_b
        cdef int c = params.p_c

        cdef int topn = params.p_TopN

        cdef np.ndarray[np.float64_t, ndim=1] X
        # cdef np.ndarray[long, ndim=1] Y

        # cdef np.ndarray[np.float64_t, ndim=1] itemlist2
        # itemlist2.resize(N)
        cdef vector[par] itemlist
        itemlist.resize(N)

        for i in xrange(N):
            itemlist[i].first = 0.0
            itemlist[i].second = 0

        oid = open(params.p_res_log_f_loc, 'a')
        oid.write(str(params.p_D))

        user_item_map = dict()
        
        for update_iter in xrange(c + 1):
            print "Iteration: " + str(update_iter)
            print str(len(data.ratings)) + " " + str(len(data.ratings) * (a + b * c))
            processing_order = range(int(len(data.ratings) * (a + b * update_iter)))

            rid = open(params.p_res_log_f_loc+str(update_iter), 'a')
            rid2 = open(params.p_res_log_f_loc+'_user', 'a') 
            # Training Phase
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

                    if not user in user_item_map:
                        user_item_map[user] = {item:1}
                    elif not item in user_item_map[user]:
                        user_item_map[user][item] = 1

                # Update parameters
                sgd_gamma *= params.p_step_dec

                rmse_err = math.sqrt(rmse_err / len(data.ratings))
                print str(tr_iter) + " " + str(rmse_err)
            
                # convergence check
                if math.fabs(rmse_err - last_rmse_err) <= params.p_conv_thres:
                    break
                last_rmse_err = rmse_err

            end = time.clock()
            print "Training Time: " + str(end - start)
            oid.write(' ' + str(end - start))

            # Getting top-N recommendation for every user
            start = time.clock()
            # itemlist2 = np.empty(N, dtype=np.float64)
            # X = P.dot(Q.T)
            for i in xrange(M):
                if update_iter == 0:
                    for j in xrange(D):
                        if j == 0:
                            rid2.write(str(P[i][j]))
                        else:
                            rid2.write(' ' + str(P[i][j])) 
                    rid2.write('\n')
                X = P[i, ].dot(Q.T)

                for j in xrange(N):
                    itemlist[j].first = X[j]
                    itemlist[j].second = j
                partial_sort(itemlist.begin(), itemlist.begin()+topn, itemlist.end(), comp_last())
                cur_topn = 0
                for j in xrange(N):
                      if (not i in user_item_map) or (not itemlist[j].second in user_item_map[i]):
                          if cur_topn == 0:
                              rid.write(str(itemlist[j].second))
                          else:
                              rid.write(' ' + str(itemlist[j].second))
                          cur_topn += 1
                          if cur_topn == topn:
                              break
                rid.write('\n')

               #  Y = (-X).argsort()
               #  cur_topn = 0
               #  for j in xrange(N):
                  #  if (not i in user_item_map) or (not Y[j] in user_item_map[i]):
                   #     if cur_topn == 0:
                    #        rid2.write(str(Y[j]))
                     #   else:
                      #      rid2.write(' ' + str(Y[j]))
                      #  cur_topn += 1
                      #  if cur_topn == topn:
                      #      break
               #  rid2.write('\n')
                    # itemlist[j] = P[i, ].dot(Q[j, ])
                #    itemlist[j] = X[j]
                
                #partial_sort(itemlist.begin(), itemlist.begin()+topn, itemlist.end())
                # bn.partsort(X[i, ], 10)

            end = time.clock()
            print "Top-N Time: " + str(end - start)
            oid.write(' ' + str(end - start))

            rid.close()
            rid2.close()
        oid.write('\n')
        oid.close()
        
