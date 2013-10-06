'''
This implements the standard SGD algorithm
'''

from params import Params
from latent_model import LatentModel
from mm_data_file import MMDataFile
from sgd_mf_machine import SGDMachine


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
    program.train(params)
    print "Model training complete..."

    # Cleanup
