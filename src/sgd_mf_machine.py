'''
Matrix factorization based on SGD
'''

import numpy as np
from mf_machine import MFMachine
import random
import sys
import math
import time


class SGDMachine(MFMachine):
    def __init__(self):
        MFMachine.__init__(self)

    def train(self, params, model, data):
        np.seterr(all='raise')

        sgd_gamma = params.p_gamma
        sgd_lambda = params.p_lambda

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

                try:
                    err = obsv - model.predict(user, item)
                    rmse_err += err * err
                    model.P[user, ] += sgd_gamma * (err * model.Q[item, ]
                                                    - sgd_lambda * model.P[user, ])
                    model.Q[item, ] += sgd_gamma * (err * model.P[user, ]
                                                    - sgd_lambda * model.Q[item, ])
                except Exception, e:
                    print e
                    print user
                    print item
                    print obsv
                    print err
                    print model.P[user, ]
                    print model.Q[item, ]
                    sys.exit("Numerical errors met!!!")

            # Update parameters
            sgd_gamma *= params.p_step_dec

            # Possible convergence check
            print "Iteration Time: " + str(time.clock() - start)
            print math.sqrt(rmse_err / len(data.ratings))
