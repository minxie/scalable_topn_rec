#!/bin/bash

python src/sgd.py -d 10 -m 95526 -n 3561 -l 1e-5 -g 1e-5 -tr "data/smallnetflix_mm" -te "data/xx.te"
