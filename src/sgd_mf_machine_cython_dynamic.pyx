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

from topn_buffer import TopNBuffer
from topn_buffer import LowerBound, UpperBound


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
        cdef double theta = params.p_theta

        cdef np.ndarray[np.float64_t, ndim=1] X

        cdef np.ndarray[np.float64_t, ndim=1] max_val
        cdef np.ndarray[np.float64_t, ndim=1] min_val
        cdef np.ndarray[np.float64_t, ndim=1] max_val_delta
        cdef np.ndarray[np.float64_t, ndim=1] min_val_delta

        cdef np.ndarray[np.float64_t, ndim=2] tmp_Q
        tmp_Q = np.random.random_sample((params.p_N, params.p_D))

        max_val = np.random.random_sample((params.p_D))
        min_val = np.random.random_sample((params.p_D))
        max_val_delta = np.random.random_sample((params.p_D))
        min_val_delta = np.random.random_sample((params.p_D))
        
        max_val.fill(-10000)
        min_val.fill(10000)
        max_val_delta.fill(-10000)
        min_val_delta.fill(10000)

        # cdef np.ndarray[np.float64_t, ndim=1] itemlist
        cdef vector[double] itemlist
        itemlist.resize(N)

        oid = open(params.p_res_log_f_loc, 'a')
        oid.write(str(params.p_D))

        t_buf = TopNBuffer(M, D, topn)
        user_buf_map = dict()
        cur_buf = 0

        top1_user_map = dict()
        user_top1_map = dict()

        total_n_buffers = 0
        sum_delta = 0
        
        for update_iter in xrange(c + 1):
            print "Iteration: " + str(update_iter)
            print str(len(data.ratings)) + " " + str(len(data.ratings) * (a + b * c))
            processing_order = range(int(len(data.ratings) * (a + b * update_iter)))
            
            new_n_buffers = 0

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
                    max_val = np.maximum(max_val, Q[item])
                    min_val = np.minimum(min_val, Q[item])

                if update_iter == 0:
                    np.copyto(tmp_Q, Q)
                else:
                    tmp_matrix = Q - tmp_Q
                    max_val_delta = np.amax(tmp_matrix, 0)
                    min_val_delta = np.amin(tmp_matrix, 0)
                    
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

            # Building the KD-Tree Index
            start = time.clock()
            end = time.clock()
            oid.write(' ' + str(end - start))
            
            # Building the M-Tree Index
            start = time.clock()
            end = time.clock()
            oid.write(' ' + str(end - start))

            # Getting top-N recommendation for every user using sort
            sum_time_sort = 0
            sum_time_kdtr = 0
            sum_time_mtre = 0
            
            for i in xrange(M):
                # print i
                
                if True:
                    start = time.clock()
                    reuse_flag = False
                    if (update_iter != 0) and (user_buf_map[i] != -1):
                        map_id = user_buf_map[i]
                        my_p_ref_pt = t_buf.p_ref_pt[map_id]                        
                        if (my_p_ref_pt - theta <= P[i]).all() and (P[i] <= my_p_ref_pt + theta).all():
                            my_buf = t_buf.p_Buffer[map_id]
                            my_UpperB = t_buf.p_UpperB[map_id]
                            for d in xrange(params.p_D):
                                my_UpperB += max(max_val_delta[d] * (my_p_ref_pt[d] + theta),
                                                 min_val_delta[d] * (my_p_ref_pt[d] - theta))
                            total_num = 0
                            for cand_item in my_buf:
                                if np.dot(Q[cand_item], P[i]) >= my_UpperB:
                                    total_num += 1
                            if total_num >= topn:
                                resuse_flag = True
                    end = time.clock()
                    sum_time_sort += end - start
                    sum_time_kdtr += end - start
                    sum_time_mtre += end - start

                    if not reuse_flag:
                        start = time.clock()
                        X = P[i, ].dot(Q.T)
                        for j in xrange(N):
                            itemlist[j] = X[j]
                        partial_sort(itemlist.begin(), itemlist.begin()+20, itemlist.end())
                        end = time.clock()
                        sum_time_sort += end - start
                        
                    # if (not reuse_flag) and (itemlist[0] in top1_user_map):
                    #     for cand_user in top1_user_map[itemlist[0]]:
                    #         cand_Creation_Iter = t_buf.p_Creation_Iter[user_buf_map[cand_user]]
                    #         if (update_iter == 0) or (cand_Creation_Iter == update_iter):
                    #             if (P[cand_user] - theta <= P[i]).all() and (P[i] <= P[cand_user] + theta).all():
                    #                 reuse_flag = True
                    #                 user_buf_map[i] = user_buf_map[cand_user]
                    #             elif user_top1_map[cand_user] == itemlist[0]:
                    #                 map_id = user_buf_map[cand_user]
                    #                 my_p_ref_pt = t_buf.p_ref_pt[map_id]
                    #                 if (my_p_ref_pt - theta <= P[i]).all() and (P[i] <= my_p_ref_pt + theta).all():
                    #                     my_buf = t_buf.p_Buffer[map_id]
                    #                     my_UpperB = t_buf.p_UpperB[map_id]
                    #                     for d in xrange(params.p_D):
                    #                         my_UpperB += max(max_val_delta[d] * (my_p_ref_pt[d] + theta),
                    #                                          min_val_delta[d] * (my_p_ref_pt[d] - theta))
                    #                 total_num = 0
                    #                 for cand_item in my_buf:
                    #                     if np.dot(Q[cand_item], P[i]) >= my_UpperB:
                    #                         total_num += 1
                    #                 if total_num >= topn:
                    #                     resuse_flag = True
                    #                     user_buf_map[i] = map_id
                    
                    if not reuse_flag: # Create new buffer
                        user_buf_map[i] = cur_buf
                        np.copyto(t_buf.p_ref_pt[user_buf_map[i]], P[i])
                        t_buf.p_Creation_Iter[user_buf_map[i]] = update_iter
                        l_LowerB = []

                        for c_iter in xrange(topn):
                            start = time.clock()
                            t_buf.p_Buffer[user_buf_map[i]].append(itemlist[c_iter])
                            LowerB = LowerBound(i, theta, itemlist[c_iter], D, P, Q)
                            l_LowerB.append(LowerB)
                            end = time.clock()
                            sum_time_sort += end - start
                            sum_time_kdtr += end - start
                            sum_time_mtre += end - start
                        
                            # KD-Tree Retrieve next
                            start = time.clock()
                            
                            end = time.clock()
                            sum_time_kdtr += end - start
                            # M-Tree Retrieve next
                            start = time.clock()
                        
                            end = time.clock()
                            sum_time_mtre += end - start
                        success_flag = False
                        for c_iter in xrange(topn,20):

                            if c_iter == topn:
                                if (np.abs(Q[itemlist[topn]] - Q[itemlist[topn-1]] <= 0.001)).all():
                                    break
                            
                            start = time.clock()
                            # Calculate Lowerbound Valeu for new item
                            t_buf.p_Buffer[user_buf_map[i]].append(itemlist[c_iter])
                            LowerB = LowerBound(i, theta, itemlist[c_iter], D, P, Q)
                            l_LowerB.append(LowerB)
                            # Calculate Upperbound Value
                            UpperB = UpperBound(t_buf, i, theta, itemlist[c_iter], itemlist[topn-1], D, P, Q, min_val, max_val, c_iter, topn, user_buf_map[i])
                            end = time.clock()
                            sum_time_sort += end - start
                            sum_time_kdtr += end - start
                            sum_time_mtre += end - start

                            # KD-Tree Retrieve next
                            start = time.clock()
                            
                            end = time.clock()
                            sum_time_kdtr += end - start
                            # M-Tree Retrieve next
                            start = time.clock()
                        
                            end = time.clock()
                            sum_time_mtre += end - start

                            start = time.clock()
                            total_num = 0
                            for cand_iter in xrange(len(l_LowerB)):
                                if l_LowerB[cand_iter] >= UpperB:
                                    total_num += 1
                            end = time.clock()

                            if total_num >= topn:
                                sum_delta += c_iter - topn
                                # print str(c_iter - topn)
                                success_flag = True
                                break
                                
                            if c_iter == 19:
                                start = time.clock()
                                partial_sort(itemlist.begin(), itemlist.begin()+100, itemlist.end())
                                end = time.clock()
                                sum_time_sort += end - start
                            elif c_iter == 99:
                                start = time.clock()
                                partial_sort(itemlist.begin(), itemlist.begin()+500, itemlist.end())
                                end = time.clock()
                                sum_time_sort += end - start
                            elif c_iter == 499:
                                start = time.clock()
                                partial_sort(itemlist.begin(), itemlist.begin()+1000, itemlist.end())
                                end = time.clock()                                
                                sum_time_sort += end - start
                            elif c_iter == 999:
                                start = time.clock()
                                partial_sort(itemlist.begin(), itemlist.begin()+10000, itemlist.end())
                                end = time.clock()                                
                                sum_time_sort += end - start
                        if success_flag:
                            total_n_buffers += 1
                            new_n_buffers += 1
                            cur_buf += 1

                            if itemlist[0] in top1_user_map:
                                top1_user_map[itemlist[0]].append(i)
                            else:
                                top1_user_map[itemlist[0]] = [i]
                                user_top1_map[i] = itemlist[0]
                        else:
                            t_buf.p_Buffer[user_buf_map[i]] = []
                            user_buf_map[i] = -1

                    


            print "Top-N Time: " + str(sum_time_sort) + ' ' + str(sum_time_kdtr) + ' ' + str(sum_time_mtre)
            oid.write(' ' + str(sum_time_sort))
            oid.write(' ' + str(sum_time_kdtr))
            oid.write(' ' + str(sum_time_mtre))
            oid.write(' ' + str(total_n_buffers))
            oid.write(' ' + str(new_n_buffers))
            oid.write(' ' + str(sum_delta))
            oid.write(' ' + str(sum_delta / total_n_buffers))

        oid.write('\n')
        oid.close()
        
