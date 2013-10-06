'''
This defines the abstract mf machine, all the interfaces
'''

import numpy as np


class MFMachine:
    def __init__(self, model):
        self._model = model

    def train(self, params):
        pass
