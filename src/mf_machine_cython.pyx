'''
This defines the abstract mf machine, all the interfaces
'''

import numpy as np


cdef class MFMachine:
    def __init__(self, model, data):
        self._model = model
        self._data = data
