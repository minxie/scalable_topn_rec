#!/bin/bash

python src/sgd.py -d 10 -m 1000 -n 1000 -l 1e-4 -g 1e-4 -tr "data/xx.tr" -te "data/xx.te"
