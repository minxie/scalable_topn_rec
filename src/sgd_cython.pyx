'''
This implements the standard SGD algorithm
'''

from params import Params
from latent_model import LatentModel
from mm_data_file import MMDataFile
from sgd_mf_machine_cython import SGDMachine
import time
import numpy as np
cimport numpy as np


def run():
    cdef np.ndarray[np.float64_t, ndim=2] P
    cdef np.ndarray[np.float64_t, ndim=2] Q

    # Parameters/Model initialization
    params = Params()
    params.parse_args("SGD method.")
    params.print_params()
    
    model = LatentModel(params)
    P = np.random.random_sample((params.p_M, params.p_D))
    Q = np.random.random_sample((params.p_M, params.p_D))
    
    # Handling I/O things
    data = MMDataFile()
    data.read_file(params)
    print "File load successful..."
    
    # Run the actual training program
    program = SGDMachine()
    start = time.clock()
    program.train(params, model, data, P, Q)
    proc_time = (time.clock() - start)
    
    print "Model training complete... Time: " + str(proc_time)

    # Cleanup
