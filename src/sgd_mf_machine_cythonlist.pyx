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
ctypedef vector[par].iterator piter
 
cdef extern from "<algorithm>" namespace "std":
    void partial_sort(diter, diter, diter, comp_last)
    void sort(piter, piter, comp_last)
    void sort(apiter, apiter, comp_last)
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
        cdef double thres = 1.0
        cdef double bound = 0.0
        cdef double dtmp = 0.0
        cdef par partmp
        cdef vector[par] Buffer

        cdef int D = params.p_D
        cdef int M = params.p_M
        cdef int N = params.p_N

        cdef double a = params.p_a
        cdef double b = params.p_b
        cdef int c = params.p_c
        
        cdef vector[vector[par]] Lists
        
        partmp.first = 0.0
        partmp.second = 0
        for i in xrange(N):
            Buffer.push_back(partmp)
        for i in xrange(D):
            Lists.push_back(Buffer)

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
            rid3 = open(params.p_res_log_f_loc+'_lst'+str(update_iter), 'a')       
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
           

            for i in xrange(N):
                for j in xrange(D):
                    Lists[j][i].first = Q[i][j]
                    Lists[j][i].second = i
            # Sorting D lists
            start = time.clock()
            for i in xrange(D):
                sort(Lists[i].begin(), Lists[i].end(), comp_last())
            end = time.clock()
            print "Sorting time: " + str(end - start)
            oid.write(' ' + str(end - start))


            # Find topk items
            start = time.clock()
            for i in xrange(M):
                #Buffer.clear()
                maxval = -10000
                notfound = True
                pointer = 0
                while notfound:
                    thres = 0.0
                    for j in xrange(D):
                        if P[i][j] > 0:
                            cur_item = pointer
                        else:
                            cur_item = N-pointer-1
                        thres += Lists[j][cur_item].first * P[i][j]
                        
                        curval = P[i, ].dot(Q[Lists[j][cur_item].second, ])
                        if curval >= maxval:
                            maxval = curval
                        #fflag = False
                        #for z in xrange(Buffer.size()):
                        #    if Buffer.at(z).second == Lists[j][cur_item].second:
                        #        fflag = True

                        #if fflag == False:
			#if fflag == True:
                        #    dtmp = 0.0
                        #    for k in xrange(D):
                        #        flag = True
                        #        itmp = 0
                        #        while flag:
                        #            if Lists[k][itmp].second == cur_item:
                        #                flag = False
                        #            else:
                        #                itmp += 1
                        #        dtmp += Lists[k][itmp].first
                        #    partmp.first = dtmp
                        #    partmp.second = cur_item
                        #    Buffer.push_back(partmp)
                           
                            # sort(Buffer.begin(), Buffer.end(), comp_last())
                            
                        #   if Buffer.size() > topn:
                        #        Buffer.pop_back()
                             
                    #bsize = Buffer.size()
                    #bound = Buffer.at(bsize-1).first
                    # print "User: " + str(i) + " Threshold: " + str(thres)
                    #if (thres <= bound) and (Buffer.size() == topn):
                    #print str(thres) + ":" + str(maxval)
                    if maxval >= thres - 0.1:    
                         notfound = False
                    else:
                        pointer += 1
                print str(i) + ' ' + str(pointer)
                #for j in xrange(D):
                #    if j == 0:
                #        rid3.write(str(Buffer[j].second))
                #    else:
                #        rid3.write(' '+ str(Buffer[j].second))
                #rid3.write('\n')
            end = time.clock()
            print "TopN Time with lists: " + str(end - start)
            oid.write(' ' + str(end - start))

            # Getting top-N recommendation for every user
            start = time.clock()

            for i in xrange(M):
                #if update_iter == 0:
                #    for j in xrange(D):
                #        if j == 0:
                #            rid2.write(str(P[i][j]))
                #        else:
                #            rid2.write(' ' + str(P[i][j])) 
                #    rid2.write('\n')
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

            end = time.clock()
            print "Top-N Time: " + str(end - start)
            oid.write(' ' + str(end - start))

            rid.close()
            rid2.close()
            rid3.close()
        oid.write('\n')
        oid.close()
        

