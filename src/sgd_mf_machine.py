'''
Matrix factorization based on SGD
'''

import numpy as np
from mf_machine import MFMachine
from random import shuffle
import sys


class SGDMachine(MFMachine):
    def __init__(self, model, data):
        MFMachine.__init__(self, model, data)

    def train(self, params):
        np.seterr(all='raise')

        sgd_gamma = params.p_gamma
        sgd_lambda = params.p_lambda

        processing_order = range(len(self._data.ratings))
        for tr_iter in xrange(params.p_max_i):
            # Shuffling order of processing the tuples
            shuffle(processing_order)

            # Run through all training examples
            rmse_err = 0.0
            for idx in processing_order:
                user = self._data.ratings[idx][0]
                item = self._data.ratings[idx][1]
                obsv = self._data.ratings[idx][2]
                err = 0.0

                try:
                    err = obsv - self._model.predict(user, item)
                    rmse_err += err * err

                    self._model.P[user, ] += sgd_gamma * (err * self._model.Q[item, ]
                                                          - sgd_lambda * self._model.P[user, ])
                    self._model.Q[item, ] += sgd_gamma * (err * self._model.P[user, ]
                                                          - sgd_lambda * self._model.Q[item, ])
                except Exception:
                    print user
                    print item
                    print obsv
                    print err
                    print self._model.P[user, ]
                    print self._model.Q[item, ]
                    sys.exit("Numerical erros met!!!")

            # Update parameters
            sgd_gamma *= params.p_step_dec

            # Possible convergence check
            print rmse_err
