#!/bin/bash

python src/run_sgd.py -d 50 -m 95526 -n 3561 -l 1e-4 -g 1e-4 -tr "data/smallnetflix_mm" -te "data/xx.te"
