from mf_machine_cython cimport MFMachine
cimport numpy as np

cdef class SGDMachine(MFMachine):
    cdef void train(self, params, model, data,
                    np.ndarray[np.float64_t, ndim=2] P,
                    np.ndarray[np.float64_t, ndim=2] Q)
