'''
This implements the standard SGD algorithm
'''

from params import Params
from latent_model import LatentModel
from mm_data_file import MMDataFile
from sgd_mf_machine_cython import SGDMachine
import time
import numpy as np


if __name__ == "__main__":
    # Parameters/Model initialization
    params = Params()
    params.parse_args("SGD method.")
    params.print_params()

    model = LatentModel(params)

    # Handling I/O things
    data = MMDataFile()
    data.read_file(params)
    print "File load successful..."
    
    # Run the actual training program
    program = SGDMachine(model, data)
    start = time.clock()
    program.train(params)
    (time.clock() - start)
    
    print "Model training complete... Time: " + str(proc_time)

    # Cleanup
